


def hybiss_data_2d():
    from tifffile import imread 
    from pathlib import Path 
    
    return imread(Path(__file__).parent/'hybiss_2d.tif')