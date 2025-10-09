import cv2
import numpy as np
import os
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
import matplotlib.pyplot as plt


def _to_xy2(a):
    a = np.squeeze(np.asarray(a)).astype(float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("Expected (N,2) array")
    return a

def _align_points(c1, c2, mode="right_bottom"):
    if mode == "right_bottom":
        dx = c2[:,0].max() - c1[:,0].max()
        dy = c2[:,1].max() - c1[:,1].max()
    elif mode == "left_top":
        dx = c2[:,0].min() - c1[:,0].min()
        dy = c2[:,1].min() - c1[:,1].min()
    elif mode == "centroid":
        dx = c2[:,0].mean() - c1[:,0].mean()
        dy = c2[:,1].mean() - c1[:,1].mean()
    else:
        raise ValueError("mode must be 'right_bottom', 'left_top', or 'centroid'")
    return c1 + np.array([dx, dy]), (dx, dy)


def display_image_with_contours(min_contour_area: float =200):
    # You can replace this with the actual path to your image
    image_path = input("Enter image path: ").strip().strip('"')  # Change this to your image file name
    
    #"""Extract pixel spacing from image metadata"""
    try:
        with Image.open(image_path) as img:
            if "Pixel Spacing" in img.info:
                pixel_spacing = img.info["Pixel Spacing"]
                print(f"Pixel Spacing: {pixel_spacing}")
                pixel_spacing = float(pixel_spacing)
                print(f"Type: {type(pixel_spacing)}")
             
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found!")
        print("Current directory:", os.getcwd())
        print("Available image files:")
        for file in os.listdir('.'):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                print(f"  - {file}")
        return
    
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    print(f"Image loaded successfully! Shape: {image.shape}")
    
    # Display original image
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Original Image', image)
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert the image to grayscale
    im_bw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Apply threshold
    (thresh, im_bw) = cv2.threshold(im_bw, 200, 255, 1)
    
    # Display binary/threshold image
#    cv2.namedWindow('Binary Image', cv2.WINDOW_NORMAL)
#    cv2.imshow('Binary Image', im_bw)
    
    # Find contours
    contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    print(f"Number of contours found: {len(contours)}")
    
    if len(filtered_contours) > 0:
        # Get the first contour
#        cnt = filtered_contours[0]
        
        # Draw contours on original image
        image1 = cv2.drawContours(image.copy(), filtered_contours, -1, (0, 255, 255), 3)
        
        # Extract contour coordinates
        contour_coordinates = []
        for contour in filtered_contours:
            # Each contour is a NumPy array of shape (n, 1, 2)
            for point in contour:
                x, y = point[0]  # Extract (x, y) coordinates
                contour_coordinates.append([x, y])  # Append coordinates to list
        
        # Convert the list to a NumPy array
        contour_coordinates_array = np.array(contour_coordinates)
        
        # Print the coordinates of all contour points
#        print("Contour coordinates:", contour_coordinates_array)
        print(f"Total contour points: {len(contour_coordinates_array)}")
        
          # Force conversion to float, removing any string data
        contour_coordinates_numeric = contour_coordinates_array.astype(float)
         # Now multiply with pixel spacing

        corrected_pixel_array = contour_coordinates_numeric * pixel_spacing

#        print("Corrected pixel array:")
#        print(corrected_pixel_array)
        # Draw contours with green color
#        image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
#        
#        # Display image with contours
#        cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
#        cv2.imshow('Contours', image_with_contours)
#        
#        # Display image with yellow contours
#        cv2.namedWindow('Yellow Contours', cv2.WINDOW_NORMAL)
#        cv2.imshow('Yellow Contours', image1)
#        
#        print("Images displayed!")
#        print("Press any key to close all windows...")
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        
        return corrected_pixel_array
    else:
        print("No contours found!")
        print("Try adjusting the threshold value.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return None
        
        
def calculate_hausdorff_distance(contours_coords_sim, contours_coords_MRI,
                                 out_path=None, invert_y=True, align_mode="right_bottom"):
    if contours_coords_sim is None or contours_coords_MRI is None:
        print("Error: inputs are None"); return None, None, None

    c1 = _to_xy2(contours_coords_sim)
    c2 = _to_xy2(contours_coords_MRI)

    # align SIM to MRI (to the right and down by default)
    c1_aligned, shift = _align_points(c1, c2, mode=align_mode)

    # Hausdorff
    d12 = directed_hausdorff(c1_aligned, c2)[0]
    d21 = directed_hausdorff(c2, c1_aligned)[0]
    hd  = max(d12, d21)

    # plot
    plt.figure()
    plt.plot(c1_aligned[:,0], c1_aligned[:,1], linewidth=1.5, label=f"SIM aligned ({shift[0]:.2f},{shift[1]:.2f})")
    plt.plot(c2[:,0],        c2[:,1],        linewidth=1.5, label="MRI")
    plt.scatter(c1_aligned[:,0], c1_aligned[:,1], s=4)
    plt.scatter(c2[:,0],        c2[:,1],        s=4)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    if invert_y:
        ax.invert_yaxis()
    plt.legend()
    plt.xlabel("x"); plt.ylabel("y")
    plt.title(f"Hausdorff: {hd:.3f}  (d12={d12:.3f}, d21={d21:.3f})")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return hd, d12, d21

def display_image_with_adjustable_threshold():
    """Version with adjustable threshold using trackbar"""
    image_path = "image.png"  # Change this to your image file name
    
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found!")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    def update_threshold(val):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale
        im_bw = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold with trackbar value
        (thresh, im_bw) = cv2.threshold(im_bw, val, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
        
        # Show images
        cv2.imshow('Binary Image', im_bw)
        cv2.imshow('Contours', image_with_contours)
        
        # Print number of contours
        cv2.setWindowTitle('Contours', f'Contours - Found: {len(contours)} contours')
    
    # Create windows
    cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Binary Image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Contours', cv2.WINDOW_NORMAL)
    
    # Create trackbar
    cv2.createTrackbar('Threshold', 'Binary Image', 200, 255, update_threshold)
    
    # Show original image
    cv2.imshow('Original Image', image)
    
    # Initial call
    update_threshold(200)
    
    print("Use the trackbar to adjust threshold value.")
    print("Press any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

   

if __name__ == "__main__":
    # Choose which version to run
    print("Choose version:")
    print("1. Fixed threshold (200)")
    print("2. Adjustable threshold with trackbar")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        display_image_with_adjustable_threshold()
    else:
        contours_coords_sim = display_image_with_contours()
        contours_coords_MRI = display_image_with_contours()
        distance_1_to_2, distance_2_to_1, bidirectional_distance = calculate_hausdorff_distance(contours_coords_sim, contours_coords_MRI)
        print(f"The max distance is {bidirectional_distance}")
        if contours_coords_sim is None or contours_coords_MRI is None:
            print(f"\nContour detection completed successfully!")
            print(f"Coordinates saved in 'contours_coords' variable")

