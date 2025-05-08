import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from deap import algorithms, base, creator, tools

# --- Configuración inicial ---
creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, 1.0, -1.0))  # Maximizar PSNR, SSIM, Colorfulness; Minimizar MSE
creator.create("Individual", list, fitness=creator.FitnessMulti)

# --- Funciones para los diferentes métodos ---

def gamma_correction(image, gamma):
    """Aplica corrección gamma a una imagen con límites seguros"""
    ''' gamma = np.clip(gamma, 0.1, 3.0)  # Rango seguro para gamma '''
    inv_gamma = 3.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def apply_clahe(image, clip_limit=2.0, grid_size=(8,8)):
    """Aplica CLAHE a una imagen con parámetros razonables"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def apply_he(image):
    """Aplica ecualización de histograma estándar"""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

def apply_homomorphic(image, gamma_h=0.5, gamma_l=1.5, cutoff=30):
    """Aplica filtrado homomórfico con parámetros por defecto razonables"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = gray.shape
    gray_float = np.float32(gray)
    dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # Crear filtro homomórfico
    mask = np.zeros((rows, cols, 2), np.float32)
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - rows/2)**2 + (j - cols/2)**2)
            mask[i,j] = (gamma_h - gamma_l) * (1 - np.exp(-(dist**2)/(cutoff**2))) + gamma_l

    filtered = dft_shift * mask
    idft_shift = np.fft.ifftshift(filtered)
    idft = cv2.idft(idft_shift)
    result = cv2.magnitude(idft[:,:,0], idft[:,:,1])
    normalized = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)

# --- Métricas de evaluación ---

def colorfulness(image):
    """Calcula la riqueza de color con manejo de casos límite"""
    ''' if len(image.shape) == 2:  # Si es escala de grises
        return 0 '''
    R, G, B = cv2.split(image.astype(np.float32))
    rg = R - G
    yb = 0.5 * (R + G) - B
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    return np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

def calculate_psnr(original, processed):
    """Calcula PSNR con protección contra divisiones por cero"""
    mse_value = mse(original, processed)
    if mse_value == 0:
        return float('inf')
    return 10 * np.log10((255**2) / mse_value)

def evaluate(individual, original_image):
    """Función de evaluación con rangos controlados"""
    gamma = individual[0]
    processed_image = gamma_correction(original_image, gamma)
    psnr = calculate_psnr(original_image, processed_image)
    ssim_val = ssim(original_image, processed_image, channel_axis=-1, win_size=7)
    colorfulness_val = colorfulness(processed_image)
    mse_val = mse(original_image, processed_image)
    return psnr, ssim_val, colorfulness_val, mse_val  # Retorna 4 objetivos

def setup_nsga2(original_image, population_size=50):
    """Configuración del algoritmo NSGA-II con rangos adecuados"""
    toolbox = base.Toolbox()
    # Rango de gamma basado en literatura: 0.5-2.5 es el rango más efectivo
    toolbox.register("attr_gamma", np.random.uniform, 0.5, 2.5)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_gamma, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, original_image=original_image)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    # Mutación centrada alrededor de 1.0 (gamma neutro)
    toolbox.register("mutate", tools.mutGaussian, mu=1.0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    return toolbox

def print_metrics(name, original, processed):
    """Imprime métricas con cálculos robustos"""
    original_float = original.astype(np.float32)
    processed_float = processed.astype(np.float32)

    psnr = calculate_psnr(original_float, processed_float)
    ssim_val = ssim(original_float, processed_float,
                   channel_axis=-1 if len(processed.shape)==3 else None,
                   win_size=7, data_range=255)
    color_val = colorfulness(processed)
    mse_val = mse(original_float, processed_float)

    print(f"{name}:")
    print(f"  PSNR = {psnr:.2f} dB")
    print(f"  SSIM = {ssim_val:.4f}")
    print(f"  Colorfulness = {color_val:.2f}")
    print(f"  MSE = {mse_val:.2f}\n")

# --- Función principal ---
def main():
    # Cargar imagen
    original_image = cv2.imread('lenna.jpg')  # Cambiar ruta de imagen
    if original_image is None:
        raise FileNotFoundError("No se pudo cargar la imagen.")

    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 1. Optimización con NSGA-II para Gamma Correction
    print("Optimizando parámetro gamma con NSGA-II...")
    toolbox = setup_nsga2(original_image)
    population = toolbox.population(n=50)

    algorithms.eaMuPlusLambda(
        population, toolbox, mu=50, lambda_=100,
        cxpb=0.7, mutpb=0.3, ngen=30, verbose=False
    )

    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    best_solution = pareto_front[np.argmax([ind.fitness.values[0] + ind.fitness.values[1] for ind in pareto_front])]
    best_gamma = np.clip(best_solution[0], 0.5, 2.5)  # Asegurar gamma en rango razonable

    # Aplicar todos los métodos
    enhanced_gamma = gamma_correction(original_image, best_gamma)
    enhanced_clahe = apply_clahe(original_image)
    enhanced_he = apply_he(original_image)
    enhanced_homo = apply_homomorphic(original_image)

    # Convertir a RGB para visualización
    enhanced_gamma_rgb = cv2.cvtColor(enhanced_gamma, cv2.COLOR_BGR2RGB)
    enhanced_clahe_rgb = cv2.cvtColor(enhanced_clahe, cv2.COLOR_BGR2RGB)
    enhanced_he_rgb = cv2.cvtColor(enhanced_he, cv2.COLOR_BGR2RGB)
    enhanced_homo_rgb = cv2.cvtColor(enhanced_homo, cv2.COLOR_BGR2RGB)

    # Calcular y mostrar métricas
    print("\n--- Métricas de Calidad ---")
    print_metrics("Original", original_image, original_image)
    print_metrics(f"NSGA-II + Gamma (γ={best_gamma:.2f})", original_image, enhanced_gamma)
    print_metrics("CLAHE", original_image, enhanced_clahe)
    print_metrics("Ecualización de Histograma (HE)", original_image, enhanced_he)
    print_metrics("Filtrado Homomórfico", original_image, enhanced_homo)

    # Visualización
    plt.figure(figsize=(15, 8))

    methods = [
        ("Original", original_image_rgb),
        (f"NSGA-II + Gamma (γ={best_gamma:.2f})", enhanced_gamma_rgb),
        ("CLAHE", enhanced_clahe_rgb),
        ("Ecualización de Histograma", enhanced_he_rgb),
        ("Filtrado Homomórfico", enhanced_homo_rgb)
    ]

    for i, (title, img) in enumerate(methods, 1):
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

main()