from PIL import Image


def convert_pgm_to_jpg(pgm_file_path, jpg_file_path):
    try:
        # Open the PGM file
        with Image.open(pgm_file_path) as img:
            # Check if the image is in 'L' mode (grayscale), if not convert it
            if img.mode != 'L':
                img = img.convert('L')

            # Save the image as a JPEG file
            img.save(jpg_file_path, 'JPEG')

        print(f"Successfully converted {pgm_file_path} to {jpg_file_path}")
    except IOError as e:
        print(f"Error opening or processing the file: {e}")


pgm_file = 'C:\Users\palap\Downloads\archive (2)\all-mias\mbd31.jpg'
jpg_file = './images/mbd31.jpg'
convert_pgm_to_jpg(pgm_file, jpg_file)
