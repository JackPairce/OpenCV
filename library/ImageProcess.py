from Requirement import *
from Types import *



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
    centers = centers.astype(np.uint8)

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
    couleur_arrière_plan = (255, 255, 255)

    # Appliquer la rotation à l'image avec les nouvelles dimensions et la couleur de fond
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (rotated_width, rotated_height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=couleur_arrière_plan,
    )

def ImageProcessing(
    image_path: str, angle: float = 0
) -> Tuple[MatLike, MatLike, MatLike, MatLike, MatLike, Sequence[Any]]:
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

    return result, sharp_image, segment, binary_image, image, contours
