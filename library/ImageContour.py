from Requirement import *
from Types import *



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
