import numpy as np
import cv2
import warnings
from skimage import measure

class Nucleus:
    
    def __init__(self, nuc_center_image, nuc_center_mask):
        self.nuc_center_mask = nuc_center_mask.astype(np.uint8)
        self.nuc_center_image = nuc_center_image.astype(np.uint8)
    
    def get_bounding_box_cords(self,single_channel_mask):
        """Single channel feature
        """
        nonzero_indices = np.nonzero(single_channel_mask)
        
        if len(nonzero_indices[0]) == 0 or len(nonzero_indices[1]) == 0:
            print("Zero bounding box co-ordinates")
            # Handle the case where the array is empty
            return [0, 0, 0, 0]
        min_row, min_col = np.min(nonzero_indices[0]), np.min(nonzero_indices[1])
        max_row, max_col = np.max(nonzero_indices[0]), np.max(nonzero_indices[1])
        return [min_row, min_col, max_row, max_col]
        
    def create_nucleus_rgb(self):
        """Extracts the RGB portion wihtin the provided contour 
        """
        
        nuc_center_mask_3ch = np.expand_dims(self.nuc_center_mask,2)
        nuc_center_mask_3ch = np.concatenate((nuc_center_mask_3ch,nuc_center_mask_3ch,nuc_center_mask_3ch), axis = 2)
        nucleus_rgb = nuc_center_mask_3ch*self.nuc_center_image
        
        return nucleus_rgb
    
    def extract_ring(self, iterations):
        """Extracts a single channel ring around the nucleus
        """
        if iterations == 0:
            raise ValueError('Number of iterations cannot be zero')
        
        A = self.nuc_center_mask
        B = self.create_dilated_mask(iterations)
        # Assuming A and B are NumPy arrays of the same shape
        mask = (A != 0)
        B[mask] = 0
        
        return B
    
    def extract_ring_rgb(self, iterations):
        """Extracts a RGB ring around the nucleus
        """
        if iterations == 0:
            raise ValueError('Number of iterations cannot be zero')
        
        ring = self.extract_ring(iterations)
        ring_cnt = measure.find_contours(ring, 0.8)
        #ring_cnt =  ring_cnt[0].astype(np.int32)
        ring_3ch = np.expand_dims(ring,2)
        ring_3ch = np.concatenate((ring_3ch,ring_3ch,ring_3ch), axis = 2)
        rgb_ring = ring_3ch*self.nuc_center_image
        
        return rgb_ring
    
    def create_dilated_mask(self, iterations):
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(self.nuc_center_mask, kernel, iterations=iterations)
        return dilated_mask



class FeatureExtractor_Ring:
    
    def __init__(self, Nucleus_Instance ,iterations):
        
        if iterations == 0:
            raise ValueError('Number of iterations cannot be zero')
        else:
            self._nucleus = Nucleus_Instance
            self.iterations = iterations
            self.ring_rgb = self._nucleus.extract_ring_rgb(iterations)
            self._channels_dict = self._create_RGB_dict(self.ring_rgb)
    
    def _create_RGB_dict(self,rgb_image):
        channels = {'R': rgb_image[:,:,0], 
                    'G': rgb_image[:,:,1], 
                    'B': rgb_image[:,:,2],
                   }
        return channels 
        
    def ring_intensity(self):
        """RGB feature
        """ 
        intensity_features = {}
        
        for channel_name, channel in self._channels_dict.items():
            min_key = f'{channel_name}_intensity_min_ring{self.iterations}'
            max_key = f'{channel_name}_intensity_max_ring{self.iterations}'
            mean_key = f'{channel_name}_intensity_mean_ring{self.iterations}'
    
            intensity_features[min_key] = channel[channel!=0].min()
            intensity_features[max_key] = channel[channel!=0].max()
            intensity_features[mean_key] = channel[channel!=0].mean()
        
        return intensity_features 
    
    def ring_r2b_ratio(self):
        """Optimized RGB feature calculation
        """
        delta = 7

        r_array = self._channels_dict['R'].astype(np.float128)
        b_array = self._channels_dict['B'].astype(np.float128)

        # Use array-wise operations instead of a loop
        lambda_array = np.abs((r_array + b_array - 255) / 255)
        r2b_ratio_array = (1 - lambda_array) * np.log((r_array + delta) / (b_array + delta))

        # Filter out zero values
        non_zero_indices = r2b_ratio_array != 0
        r2b_ratio_list = r2b_ratio_array[non_zero_indices]

        r2b_dict = {
            f'r2b_mean_ring{self.iterations}': np.mean(r2b_ratio_list),
            f'r2b_var_ring{self.iterations}': np.var(r2b_ratio_list)
        }

        return r2b_dict
    
    
    def ring_brightness(self):
        """RGB feature
        """
        
        hsv_nuc = cv2.cvtColor(self.ring_rgb, cv2.COLOR_RGB2HSV)
        brightness_channel = hsv_nuc[:, :, 2]

        min_ = np.min(brightness_channel)
        max_ = np.max(brightness_channel)
        brightness_channel = ((brightness_channel-min_)/(max_-min_))
    
        brightness_dict = {f'brightness_mean_ring{self.iterations}': np.mean(brightness_channel),
                           f'brightness_var_ring{self.iterations}' : np.var(brightness_channel)
                          }
        return brightness_dict 
    

class FeatureExtractor_Nucleus:
    def __init__(self, Nucleus_Instance):
        
        self._nucleus = Nucleus_Instance
        
        contours, _ = cv2.findContours(self._nucleus.nuc_center_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>1:
             warnings.warn("More than one contour found. Working with the first contour.", UserWarning)
        self.contour = contours[0]
        
        self._nucleus_rgb = self._nucleus.create_nucleus_rgb()
        self._channels_dict = self._create_RGB_dict(self._nucleus_rgb)
        
    def _create_RGB_dict(self,rgb_image):
        channels = {'R': rgb_image[:,:,0], 
                    'G': rgb_image[:,:,1], 
                    'B': rgb_image[:,:,2],
                   }
        return(channels)
    
    def nucleus_circularity(self):
        
        A = cv2.contourArea(self.contour)
        P = cv2.arcLength(self.contour, closed = True)
        circularity = (4* np.pi* A)/ (P**2)
        return({'nucleus_circularity':circularity})
    
    def nucleus_area(self):
        area = np.sum(self._nucleus.nuc_center_mask)
        return({'area':area})
    
    def get_bounding_box_cords(self):
        """Single channel feature
        """
        nonzero_indices = np.nonzero(self._nucleus.nuc_center_mask)
        min_row, min_col = np.min(nonzero_indices[0]), np.min(nonzero_indices[1])
        max_row, max_col = np.max(nonzero_indices[0]), np.max(nonzero_indices[1])
        return([min_row, min_col, max_row, max_col])
    
    def get_bounding_box_area(self):
        """Single channel feature
        """
        min_row, min_col, max_row, max_col = self.get_bounding_box_cords()
        bounding_box_area = (max_row - min_row + 1) * (max_col - min_col + 1)
        return({'bbox_area':bounding_box_area})

    def nucleus_intensity(self):
        """RGB feature"""
        intensity_features = {}

        for channel_name, channel in self._channels_dict.items():
            non_zero_indices = channel != 0
            min_key = f'{channel_name}_intensity_min'
            max_key = f'{channel_name}_intensity_max'
            mean_key = f'{channel_name}_intensity_mean'

            intensity_features[min_key] = channel[non_zero_indices].min()
            intensity_features[max_key] = channel[non_zero_indices].max()
            intensity_features[mean_key] = channel[non_zero_indices].mean()

        return intensity_features
    
    def nucleus_r2b_ratio(self):
        """RGB feature
        """
        delta = 7

        r_array = self._channels_dict['R'].astype(np.float128)
        b_array = self._channels_dict['B'].astype(np.float128)

        # Use array-wise operations instead of a loop
        lambda_array = np.abs((r_array + b_array - 255) / 255)
        r2b_ratio_array = (1 - lambda_array) * np.log((r_array + delta) / (b_array + delta))

        # Filter out zero values
        non_zero_indices = r2b_ratio_array != 0
        r2b_ratio_list = r2b_ratio_array[non_zero_indices]

        r2b_dict = {
            f'r2b_mean': np.mean(r2b_ratio_list),
            f'r2b_var': np.var(r2b_ratio_list)
        }

        return(r2b_dict)
    
    
    def nucleus_brightness(self):
        """RGB feature
        """
        hsv_nuc = cv2.cvtColor(self._nucleus_rgb, cv2.COLOR_RGB2HSV)
        brightness_channel = hsv_nuc[:, :, 2]

        min_ = np.min(brightness_channel)
        max_ = np.max(brightness_channel)
        brightness_channel = ((brightness_channel-min_)/(max_-min_))
    
        brightness_dict = {'brightness_mean': np.mean(brightness_channel),
                           'brightness_var' : np.var(brightness_channel)
                          }

        return (brightness_dict)
    
    def nucleus_convex_hull_area(self):
        """Single channel feature
        """
        hull = cv2.convexHull(self.contour)
        hull_area = cv2.contourArea(hull)
        return ({'convex_hull_area':hull_area})
    
    def nucleus_perimeter(self):
        """Single channel feature
        """
        perimeter = cv2.arcLength(self.contour, closed=True)
        return ({'perimeter':perimeter})
    
    def nucleus_solidity(self):
        """Single channel feature
        """
        solidity = self.nucleus_area()['area']/self.nucleus_convex_hull_area()['convex_hull_area']
        return ({'solidity':solidity})
    
    def nucleus_hu_moments(self):
        """Single channel feature
        """
        moments = cv2.moments(self._nucleus.nuc_center_mask)
        huMoments = cv2.HuMoments(moments)
        hu_moments_dict = {
            f"huMoment{i + 1}": huMoments[i][0] for i in range(len(huMoments))
        }
        
        return (hu_moments_dict)
    
    def nucleus_ellipse_properties(self):
        """Single channel feature
        """
        ellipse = cv2.fitEllipse(self.contour)
        minor_axis_length =  ellipse[1][0]
        major_axis_length =  ellipse[1][1]
        orientation = ellipse[2]
        
        return ({'major_axis_length':major_axis_length, 'minor_axis_length': minor_axis_length, 'orientation': orientation})#,'eccentricity':eccentricity})