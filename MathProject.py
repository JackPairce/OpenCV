import cv2
import numpy as np
from matplotlib import pyplot as plt, patches
import warnings

# Typing Modules
from cv2.typing import MatLike
from typing import Any, Dict, List, Optional, Tuple, Sequence
from numpy import ndarray

# Progress Bar
from tqdm import tqdm


def segment_image_kmeans(image: MatLike, k: int = 2) -> MatLike:
    """
    Segmenter une image en utilisant le clustering k-means.

    Parameters:
    - image: Image à segmenter.
    - k: Nombre de clusters (segments).

    Returns:
    - Image segmentée en niveaux de gris.
    """

    # Aplatir l'image en un tableau 1D d'échantillons
    pixels: Any = image.flatten().reshape((-1, 1))

    # Convertir les données en type float32
    pixels = np.float32(pixels)

    # Définir les critères de convergence (ajuster si nécessaire)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Appliquer la méthode k-means avec une initialisation améliorée
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )

    # Convertir les centres des clusters en entiers
    centers = np.uint8(centers)

    # Mapper chaque pixel à son cluster correspondant
    segmented_image = centers[labels.flatten()]

    # Remettre l'image dans sa forme originale
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

def SharpnessFilter(img: MatLike):
    """
    Appliquer un filtre de netteté à une image.

    Parameters:
    - img: Image à traiter.

    Returns:
    - Image filtrée.
    """
    # Définir un noyau d'affûtage
    noyau = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # Appliquer le noyau d'affûtage
    return cv2.filter2D(img, -1, noyau)

def get_contours(segmented_image: MatLike) -> Sequence[MatLike]:
    """
    Extraire les contours d'une image segmentée.

    Parameters:
    - segmented_image: Image segmentée en niveaux de gris.

    Returns:
    - Liste de contours.
    """
    noyau = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Obtenir les bords de Canny
    contours_canny = cv2.Canny(segmented_image, 1, 50)

    contours_canny = cv2.morphologyEx(contours_canny, cv2.MORPH_CLOSE, noyau)

    contours, _ = cv2.findContours(contours_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours

def rotate_image(image, angle: float):
    """
    Faire pivoter une image.

    Parameters:
    - image: Image à faire pivoter.
    - angle: Angle de rotation en degrés.

    Returns:
    - Image pivotée.
    """
    # Obtenir les dimensions de l'image
    height, width = image.shape[:2]

    # Calculer la matrice de rotation
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Calculer les nouvelles dimensions de l'image rotatée
    rotated_width = int(
        abs(rotation_matrix[0, 0] * width) + abs(rotation_matrix[0, 1] * height)
    )
    rotated_height = int(
        abs(rotation_matrix[1, 0] * width) + abs(rotation_matrix[1, 1] * height)
    )

    # Ajuster la matrice de rotation pour tenir compte de la translation
    rotation_matrix[0, 2] += (rotated_width - width) / 2
    rotation_matrix[1, 2] += (rotated_height - height) / 2

    # Spécifier la couleur de fond (blanc dans cet exemple)
    couleur_arriere_plan = (255, 255, 255)

    # Appliquer la rotation à l'image avec les nouvelles dimensions et la couleur de fond
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (rotated_width, rotated_height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=couleur_arriere_plan,
    )

def ImageProcessing(
    image_path: str, angle: float = 0
) -> List[MatLike | Sequence[Any]]:
    """
    Traiter une image et extraire les contours.

    Parameters:
    - image_path: Chemin vers le fichier image.
    - angle: Angle de rotation en degrés (par défaut 0).

    Returns:
    - Tuple contenant l'image traitée, l'image de netteté, l'image segmentée, l'image binaire,
      l'image en niveau de gris et une liste de contours.
    """

    # Charger l'image
    result = cv2.imread(image_path)

    # Vérifier si l'image est chargée avec succès
    if result is None:
        raise ValueError("Erreur : Impossible de charger l'image.")

    # Rotater l'image si nécessaire
    if angle != 0:
        # Rotater l'image
        result = rotate_image(result, angle)

    try:
        # Appliquer le filtre de netteté
        sharp_image = SharpnessFilter(result)

        # Segmenter l'image avec la méthode de k-means
        segment = segment_image_kmeans(sharp_image)

        # Convertir l'image segmentée en binaire
        _, binary_image = cv2.threshold(segment, 128, 255, cv2.THRESH_BINARY)

        # Convertir l'image binaire en niveau de gris
        image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

        # Extraire les contours de l'image
        contours = get_contours(image)
    except cv2.error as e:
        raise ValueError(f"Erreur lors de la recherche des contours : {e}")

    return [result, sharp_image, segment, binary_image, image, contours]

def filter_contours_by_perimeter(contours: Any, threshold: float = 500) -> Any:
    """
    Filtrer les contours en fonction de leur périmètre.

    Parameters:
    - contours: Liste de contours.
    - threshold: Périmètre minimum pour inclure un contour.

    Returns:
    - Le contour avec la longueur d'arc minimale parmi ceux satisfaisant la condition.

    Raises:
    - ValueError: Si aucun contour valide n'est trouvé.
    """
    valid_contours = [c for c in contours if cv2.arcLength(c, True) > threshold]

    if valid_contours:
        return min(valid_contours, key=lambda c: cv2.arcLength(c, True))
    else:
        return contours[0]

def generate_angles(start_angle=0, end_angle=2 * np.pi, num_sectors=10) -> ndarray:
    """
    Generate an array of angles evenly spaced between start_angle and end_angle.

    Parameters:
    - start_angle: Starting angle in radians.
    - end_angle: Ending angle in radians.
    - num_partitions: Number of partitions to generate.

    Returns:
    - NumPy array of angles.
    """
    return np.linspace(start_angle, end_angle, num_sectors + 1)

def EuclideanDistance(FirstPoint, SecondPoint: Tuple[float, float]):
    """Calculate the Euclidean distance between two points."""
    if isinstance(FirstPoint, np.ndarray):
        C0, C1 = FirstPoint[:, 0, 0], FirstPoint[:, 0, 1]
        return np.sqrt((C0 - SecondPoint[0]) ** 2 + (C1 - SecondPoint[1]) ** 2)
    else:
        C0, C1 = FirstPoint
        D0, D1 = SecondPoint
        return np.sqrt((C0 - D0) ** 2 + (C1 - D1) ** 2)

def Get_Center(contour: ndarray) -> Tuple[int, int]:
    M = cv2.moments(contour)
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

def CalculateTheta(Y: float, X: float) -> float:
    theta = np.arctan2(Y, X)
    if theta < 0:
        theta += 2 * np.pi
    return theta

def classify_angles(
    XY: List[float], CXY: List[float], angles: np.ndarray
) -> Tuple[Optional[str], Optional[List[float]]]:
    x, y = XY
    cx, cy = CXY
    theta = CalculateTheta((y - cy), (x - cx))
    for start, end in zip(angles[:-1], angles[1:]):
        if start <= theta < end:
            return f"{start/np.pi}π-{end/np.pi}π", XY

    return None, None

    # FP = max(enumerate(distances), key=lambda x: x[1])[0]
    # plt.plot([c[:, :, 0][FP][0], center[0]], [c[:, :, 1][FP][0], center[1]])

def plot_circle_and_sectors(
    c: ndarray, center: Tuple[float, float], num_sectors: int = 10
) -> Tuple[float, List[Tuple[float, float]]]:
    """
    Plot a circle and angular sectors around a specified center point.

    Parameters:
    - c: NumPy array representing points.
    - center: Coordinates of the center point.
    - num_sectors: Number of angular sectors around the center.

    Returns:
    None (plots the circle and sectors).
    """
    circle_radius = max(EuclideanDistance(c, center))

    # circle = patches.Circle(center, circle_radius, color="blue", fill=False)
    # plt.gca().add_patch(circle)

    angles = generate_angles(num_sectors=num_sectors)
    END = []
    for angle in angles:
        end_x = int(center[0] + circle_radius * np.cos(angle))
        end_y = int(center[1] + circle_radius * np.sin(angle))
        END.append((end_x, end_y))

    return circle_radius, END
    # ax.plot([center[0], end_x], [center[1], end_y], color="green")

def process_points(
    c: np.ndarray, CXY: List[float], angles: np.ndarray
) -> Dict[str, List[List[float]]]:
    """
    Process points from the array 'c' and classify them based on the given angles.

    Parameters:
    - c: 3D NumPy array representing points.
    - cx, cy: Coordinates of the center point.
    - angles: NumPy array containing angle intervals.

    Returns:
    - Dictionary where keys are angle intervals and values are lists of classified points.
    """
    R = {}
    cx, cy = CXY
    for x, y in zip(c[:, 0, 0], c[:, 0, 1]):
        interval, value = classify_angles([x, y], [cx, cy], angles)
        R.setdefault(interval, []).append(value)

    return {
        i: sorted(
            v, key=lambda x: CalculateTheta((x[1] - cy), (x[0] - cx)), reverse=True
        )
        for i, v in R.items()
    }

def generate_newton_polynomial(x, y):
    """
    Generate the Newton polynomial function and calculate the error for interpolation.

    Parameters:
    - x: List or array of x-values.
    - y: List or array of y-values corresponding to x.

    Returns:
    - Tuple (newton_poly, errors), where newton_poly is the Newton polynomial function
      and errors is a list of absolute errors for each point in x.
    """
    n = len(x)
    F = np.zeros((n, n))
    F[:, 0] = y
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for j in range(1, n):
            for i in range(j, n):
                F[i, j] = (F[i, j - 1] - F[i - 1, j - 1]) / (x[i] - x[i - j])

        coef = F.diagonal()

        def newton_polynomial(x_data: ndarray) -> Tuple[ndarray, float]:
            n = len(x) - 1
            result = coef[n]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                for k in range(1, n + 1):
                    result = result * (x_data - x[n - k]) + coef[n - k]

            interpolated_value = result
            actual_value = np.interp(x_data, x, y)
            absolute_error = np.abs(interpolated_value - actual_value)
            if not np.any(absolute_error):
                return result, float("inf")
            return result, max(absolute_error)

        return newton_polynomial

def least_squares_approximation(x, y, degree=3):
    """
    Approximation au sens des moindres carrés avec un polynôme de degré donné.

    Parameters:
    - x: Liste des valeurs x.
    - y: Liste des valeurs y correspondantes.
    - degree: Degré du polynôme d'approximation.

    Returns:
    - Coefficients du polynôme d'approximation.
    """
    return np.polyfit(x, y, degree)

def get_equidistant_points(x, y, num_points: int = 5):
    """
    Get a subset of equidistant points from x and y.

    Parameters:
    - x: List or array of x-values.
    - y: List or array of y-values.
    - num_points: Number of equidistant points to select.

    Returns:
    - Tuple (x_subset, y_subset) containing the selected points.
    """
    indices = np.linspace(0, len(x) - 1, num_points, dtype=int)
    return np.array(x)[indices], np.array(y)[indices]

def GetTheBestInterpolation(
    point,
    X,
):
    PointsEvaluation = 4
    CurrentEvaluation = float("inf")
    # len(point)
    for i in range(4, 31):
        Xp, Yp = get_equidistant_points(
            [sub[0] for sub in point], [sub[1] for sub in point], i
        )
        Pn, Err = generate_newton_polynomial(Xp, Yp)(X)

        if Err < CurrentEvaluation or Err == 0:
            PointsEvaluation = i
            CurrentEvaluation = Err

    return PointsEvaluation, CurrentEvaluation

def least_squares_polynomial(
    x: List[float], y: List[float], x_values: ndarray, degree=3
) -> Tuple[np.ndarray, float]:
    """
    Perform least squares approximation with a polynomial of a given degree.
    Optionally, evaluate the polynomial at specific x-values.

    Parameters:
    - x: List or array of x-values.
    - y: List or array of y-values corresponding to x.
    - degree: Degree of the polynomial approximation.
    - x_values: Optional - Array of values at which to evaluate the polynomial.

    Returns:
    - If x_values is None, returns the coefficients of the polynomial.
    - If x_values is provided, returns a tuple (coefficients, results, errors),
      where coefficients are the coefficients of the polynomial,
      results are the results of the polynomial evaluation at x_values,
      and errors are the absolute errors between the interpolated values and the actual values.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=np.RankWarning)
        coefficients = np.polyfit(x, y, degree)

    results = np.polyval(coefficients, x_values)
    actual_values = np.interp(x_values, x, y)
    errors = np.abs(results - actual_values)
    if not np.any(errors):
        return results, float("inf")
    return results, max(errors)
    return results, sum(errors) / len(errors)

def ApproximationPoly(c, cx, cy, RangeStart, RangeEnd):
    NumSectors = 10
    BestPointsEvaluation = {}
    BestEvaluation = float("inf")

    TestRange = [i for i in range(RangeStart, RangeEnd + 1)]

    for _, num_sectors in zip(tqdm(TestRange, desc="ApproximationPoly"), TestRange):
        angles = generate_angles(num_sectors=num_sectors)
        R = process_points(c, [cx, cy], angles)

        PointsEvaluation = {}
        CurrentErr: List[float] = []
        for SectotStr, point in R.items():
            start, end = sorted([point[0][0], point[-1][0]])
            X = np.arange(start, end, 0.1)

            DegreeErr: List[Tuple[int, float]] = []
            for D in range(3, 40):
                _, Err = least_squares_polynomial(
                    [sub[0] for sub in point], [sub[1] for sub in point], X, degree=D
                )
                DegreeErr.append((D, Err))
            DegreeErr.sort(key=lambda x: x[1])
            CurrentErr.append(DegreeErr[0][1])
            PointsEvaluation[SectotStr] = DegreeErr[0][0]

        CurrentErrValue = max(CurrentErr)
        if CurrentErrValue < BestEvaluation:
            BestEvaluation = CurrentErrValue
            NumSectors = num_sectors
            BestPointsEvaluation = PointsEvaluation

    return BestPointsEvaluation, NumSectors

def PlotPlyFit(c, cx, cy, BestPointsEvaluation={}, NumSectors=10):
    angles = generate_angles(num_sectors=NumSectors)
    if not BestPointsEvaluation:
        BestPointsEvaluation = {
            f"{S/np.pi}π-{angles[i+1]/np.pi}π": 3 for i, S in enumerate(angles[:-1])
        }

    R = process_points(c, [cx, cy], angles)
    MoyErr = []
    Points:List[Tuple[ndarray,ndarray]] = []
    MoyErr: List[float] = []
    for SectroName, point in R.items():
        start, end = sorted([point[0][0], point[-1][0]])
        X = np.arange(start, end, 0.1)
        Pn, Err = least_squares_polynomial(
            [sub[0] for sub in point],
            [sub[1] for sub in point],
            X,
            degree=BestPointsEvaluation[SectroName],
        )
        MoyErr.append(Err)
        Points.append((X, Pn))
    return Points, max(MoyErr)
    return Points, sum(MoyErr) / len(MoyErr)

def process_evaluation(
    c: ndarray, cx: float, cy: float, num_sectors: int
) -> Tuple[Dict[str, int], float]:
    angles = generate_angles(num_sectors=num_sectors)
    R = process_points(c, [cx, cy], angles)

    PointsEvaluation = {}
    CurrentErr = []

    for SectotStr, point in R.items():
        start, end = sorted([point[0][0], point[-1][0]])
        X = np.arange(start, end, 0.1)

        Evaluation, Err = GetTheBestInterpolation(point, X)
        PointsEvaluation[SectotStr] = Evaluation
        CurrentErr.append(Err)

    return PointsEvaluation, max(CurrentErr)
    return PointsEvaluation, sum(CurrentErr) / len(CurrentErr)

def NewtonInterpolation(
    c: ndarray, cx: float, cy: float, RangeStart: int, RangeEnd: int,NumSectors = 10
) -> Tuple[Dict[str, int], int]:
    NumSectors = 10
    BestPointsEvaluation = {}
    BestEvaluation = float("inf")

    TestRange = range(RangeStart, RangeEnd + 1)
    for _, num_sectors in zip(tqdm(TestRange, desc="NewtonInterpolation"), TestRange):
        PointsEvaluation, CurrentErrValue = process_evaluation(c, cx, cy, num_sectors)

        if CurrentErrValue < BestEvaluation:
            BestEvaluation = CurrentErrValue
            NumSectors = num_sectors
            BestPointsEvaluation = PointsEvaluation

    return BestPointsEvaluation, NumSectors

def PlotPoly(
    c, cx, cy, BestPointsEvaluation={}, NumSectors=10
) -> Tuple[List[Tuple[ndarray, ndarray]], float]:
    angles = generate_angles(num_sectors=NumSectors)
    if not BestPointsEvaluation:
        BestPointsEvaluation = {
            f"{S/np.pi}π-{angles[i+1]/np.pi}π": 4 for i, S in enumerate(angles[:-1])
        }

    R = process_points(c, [cx, cy], angles)
    MoyErr = []
    Points = []
    for i, (SectroName, point) in enumerate(sorted(R.items())):
        start, end = sorted([point[0][0], point[-1][0]])
        X = np.arange(start, end, 0.1)
        Xp, Yp = get_equidistant_points(
            [sub[0] for sub in point],
            [sub[1] for sub in point],
            BestPointsEvaluation[SectroName],
        )
        Pn, Err = generate_newton_polynomial(Xp, Yp)(X)
        MoyErr.append(Err)
        Points.append((X, Pn))
    return Points, max(MoyErr)

def ThresholdSearch(contours) -> float:
    for _, i in zip(
        tqdm(range(len(contours)), desc="ThresholdSearch"),
        [cv2.arcLength(c, True) for c in contours],
    ):
        Cl = filter_contours_by_perimeter(contours, i)
        if cv2.moments(Cl)["m00"]:
            Cxl, Cyl = Get_Center(Cl)
            if min(EuclideanDistance(Cl, (Cxl, Cyl))) > 100:
                return i
    return 500

if __name__ == "__main__":
    import os

    Config = {
        3206: [267.78174459934235, -55.311890840361734],
        3283: [831.0437180995941, 165.5],
        3342: [3103.682450890541, -90],
        3549: [3756.4347579479218, 0],
    }

    Newton = {3206: 23, 3283: 42, 3342: 45, 3549: 30}
    fit = {3206: 59, 3283: 139, 3342: 190, 3549: 110}

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Les Contour", fontsize=16)
    fig.set_size_inches(12, 8)
    for i, (ax, image_path) in enumerate(
        zip(axs.flatten(), os.listdir("./Projets Maths/Images/"))
    ):
        _, _, _, _, img, contours = ImageProcessing(
            f"./Projets Maths/Images/{image_path}",
            Config[int(image_path.split(".")[0])][1],
        )

        Th = Config[int(image_path.split(".")[0])][0]
        c = filter_contours_by_perimeter(contours, Th)

        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_adjustable("box")

        Cx, Cy = Get_Center(c)

        conf, sector = ApproximationPoly(c, Cx, Cy, 23, 50)

        points, Err = PlotPlyFit(c, Cx, Cy, conf, sector)

        for X, Pn in points:
            ax.plot(X, Pn)
        print(
            f"Max Degree {max(list(conf.values()))-1}, Num Sector {sector}, Err {Err}"
        )
        ax.set_title(f"Numbre De Sector {sector}")
        ax.axis("off")
    fig.tight_layout(pad=2.0)
    plt.show()


# Max Degree 9, Num Sector 23, Err Moy 43.79117195795987
# Max Degree 14, Num Sector 42, Err Moy 15.78624342412661
# Max Degree 14, Num Sector 45, Err Moy 21.105424309506777
# Max Degree 10, Num Sector 30, Err Moy 19.582140526547352
