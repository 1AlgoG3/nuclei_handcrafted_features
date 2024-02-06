from tqdm.auto import tqdm
import pandas as pd

from handcrafted_features.feature_definition import (
    Nucleus, 
    FeatureExtractor_Ring, 
    FeatureExtractor_Nucleus, 
)


def extract_ring_fe_object(nucleus_object, iterations):
    ring_objects = [FeatureExtractor_Ring(nucleus_object, iteration) for iteration in iterations]
    return ring_objects

def extract_nucleus_objects(hcfpd):
    
    nucleus_objects = []
    
    for nucleus in tqdm(hcfpd):
        nucleus_objects.append(Nucleus(nucleus['nuc_center_image'], nucleus['nuc_center_mask']))
    
    return(nucleus_objects)

def extract_fe_objects(nucleus_objects,iterations):
    
    nucleus_fe_objects = []
    ring_fe_objects = []
    
    for nucleus_object in tqdm(nucleus_objects):
        nucleus_fe_objects.append(FeatureExtractor_Nucleus(nucleus_object))
        ring_fe_objects.append(extract_ring_fe_object(nucleus_object,iterations))
        
    return(nucleus_fe_objects, ring_fe_objects)

class FeatureExtractor:
    
    def __init__(self, hf_folder, nucleus_fe_objects, ring_fe_objects):
        self._hf_folder = hf_folder
        self._nucleus_fe_objects = nucleus_fe_objects
        self._ring_fe_objects = ring_fe_objects
    
    def nucleus_area(self):
        
        nucleus_area = []
        print('nucleus_area')
        for nucleus_fe_object in tqdm(self._nucleus_fe_objects):
            nucleus_area.append(nucleus_fe_object.nucleus_area())
    
        print('Saving file')
        nucleus_area_path = f'{self._hf_folder}/nucleus_area.csv'
        pd.DataFrame(nucleus_area).to_csv(nucleus_area_path,index = False)
        print('complete')
        print(f'File saved at: {nucleus_area_path}')
    
    def nucleus_brightness(self):
        
        nucleus_brightness = []
        print('nucleus_brightness')
        for nucleus_fe_object in tqdm(self._nucleus_fe_objects):
            nucleus_brightness.append(nucleus_fe_object.nucleus_brightness())
    
        print('Saving file')
        nucleus_brightness_path = f'{self._hf_folder}/nucleus_brightness.csv'
        pd.DataFrame(nucleus_brightness).to_csv(nucleus_brightness_path,index = False)
        print('complete')
        print(f'File saved at: {nucleus_brightness_path}')
    
    def nucleus_circularity(self):

        nucleus_circularity = []
        print('nucleus_circularity')
        for nucleus_fe_object in tqdm(self._nucleus_fe_objects):
            nucleus_circularity.append(nucleus_fe_object.nucleus_circularity())
    
        print('Saving file')
        nucleus_circularity_path = f'{self._hf_folder}/nucleus_circularity.csv'
        pd.DataFrame(nucleus_circularity).to_csv(nucleus_circularity_path,index = False)
        print('complete')
        print(f'File saved at: {nucleus_circularity_path}')
        
    def nucleus_convex_hull_area(self):

        nucleus_convex_hull_area =[]
        print('nucleus_convex_hull_area')
        for nucleus_fe_object in tqdm(self._nucleus_fe_objects):
            nucleus_convex_hull_area.append(nucleus_fe_object.nucleus_convex_hull_area())
    
        print('Saving file')
        nucleus_convex_hull_area_path = f'{self._hf_folder}/nucleus_convex_hull_area.csv'
        pd.DataFrame(nucleus_convex_hull_area).to_csv(nucleus_convex_hull_area_path,index = False)
        print('complete')
        print(f'File saved at: {nucleus_convex_hull_area_path}')
    
    def nucleus_hu_moments(self):

        nucleus_hu_moments =[]
        print('nucleus_hu_moments')
        for nucleus_fe_object in tqdm(self._nucleus_fe_objects):
            nucleus_hu_moments.append(nucleus_fe_object.nucleus_hu_moments())
    
        print('Saving file')
        nucleus_hu_moments_path = f'{self._hf_folder}/nucleus_hu_moments.csv'
        pd.DataFrame(nucleus_hu_moments).to_csv(nucleus_hu_moments_path,index = False)
        print('complete')
        print(f'File saved at: {nucleus_hu_moments_path}')
    
    def nucleus_intensity(self):

        nucleus_intensity =[]
        print('nucleus_intensity')
        for nucleus_fe_object in tqdm(self._nucleus_fe_objects):
            nucleus_intensity.append(nucleus_fe_object.nucleus_intensity())
    
        print('Saving file')
        nucleus_intensity_path = f'{self._hf_folder}/nucleus_intensity.csv'
        pd.DataFrame(nucleus_intensity).to_csv(nucleus_intensity_path,index = False)
        print('complete')
        print(f'File saved at: {nucleus_intensity_path}')
    
    def nucleus_r2b_ratio(self):

        nucleus_r2b_ratio =[]
        print('nucleus_r2b_ratio')
        for nucleus_fe_object in tqdm(self._nucleus_fe_objects):
            nucleus_r2b_ratio.append(nucleus_fe_object.nucleus_r2b_ratio())
    
        print('Saving file')
        nucleus_r2b_ratio_path = f'{self._hf_folder}/nucleus_r2b_ratio.csv'
        pd.DataFrame(nucleus_r2b_ratio).to_csv(nucleus_r2b_ratio_path,index = False)
        print('complete')
        print(f'File saved at: {nucleus_r2b_ratio_path}')
    
    def nucleus_solidity(self):
        
        nucleus_solidity =[]
        print('nucleus_solidity')
        for nucleus_fe_object in tqdm(self._nucleus_fe_objects):
            nucleus_solidity.append(nucleus_fe_object.nucleus_solidity())

        print('Saving file')
        nucleus_solidity_path = f'{self._hf_folder}/nucleus_solidity.csv'
        pd.DataFrame(nucleus_solidity).to_csv(nucleus_solidity_path,index = False)
        print('complete')
        print(f'File saved at: {nucleus_solidity_path}')
        
    def ring_brightness(self):
        
        ring_brightness = []
        print('ring_brightness')
        for rings in tqdm(self._ring_fe_objects):
            feature = {}
            for ring in rings:
                feature.update(ring.ring_brightness())
            ring_brightness.append(feature)

        print('Saving file')
        ring_brightness_path = f'{self._hf_folder}/ring_brightness.csv'
        pd.DataFrame(ring_brightness).to_csv(ring_brightness_path,index = False)
        print('complete')
        print(f'File saved at: {ring_brightness_path}')
        
    def ring_intensity(self):

        ring_intensity = []
        print('ring_intensity')
        for rings in tqdm(self._ring_fe_objects):
            feature = {}
            for ring in rings:
                feature.update(ring.ring_intensity())
            ring_intensity.append(feature)

        print('Saving file')
        ring_intensity_path = f'{self._hf_folder}/ring_intensity.csv'
        pd.DataFrame(ring_intensity).to_csv(ring_intensity_path,index = False)
        print('complete')
        print(f'File saved at: {ring_intensity_path}')
    
    def ring_r2b_ratio(self):
        
        ring_r2b_ratio = []
        print('ring_r2b_ratio')
        for rings in tqdm(self._ring_fe_objects):
            feature = {}
            for ring in rings:
                feature.update(ring.ring_r2b_ratio())
            ring_r2b_ratio.append(feature)

        print('Saving file')
        ring_r2b_ratio_path = f'{self._hf_folder}/ring_r2b_ratio.csv'
        pd.DataFrame(ring_r2b_ratio).to_csv(ring_r2b_ratio_path,index = False)
        print('complete')
        print(f'File saved at: {ring_r2b_ratio_path}')
        