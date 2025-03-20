
# numpy for obv reasons bro
import numpy as np

# cv2 for glaussian blur
import cv2

# rasterio to deal with raster data
'''
what is raster data you ask?
raster data is like a 2d grid of data(like a photo) whwere each pixel stores somekind of value such as temperature,elevtion, colour
rasterio is used often with satelite imagery where tif files are used to store the raster data.
'''
import rasterio
from rasterio.warp import transform_bounds

# to specify path for the 6s model
import os
os.environ["SIXS_EXECUTABLE"] = "/full/path/to/6S_executable"

# to use the following methods from the py6s library to run the 6s model
from Py6S import SixS, AtmosProfile, AeroProfile, GroundReflectance, Wavelength

# to make requests for dem data, aot, etc from sentinel hub
import requests

# to use the warp function from gdal to correct for terrain distortions
from osgeo import gdal

# to specify folder paths
from pathlib import Path


# class containing all relevant functionalities related to radiometric corrections
class RadiometricCorrection:
    # the initializer method for this class which sets up the following attributes on obj creation
    def __init__(self, kernel_size = 11, scalefactor = 0.0001):
        self.dark_frame = np.random.normal(loc=0, scale=0.5, size=(64, 64)).astype(np.float32)
        self.kernel_size = kernel_size
        self.scalefactor = scalefactor
    
    # function to find the flat_field
    def estimate_flat_field(self,image):
        # Convert to float for more precise calculations than int
        float_image = image.astype(np.float32)
        
        """Apply Gaussian smoothing to estimate illumination pattern.
          It works by convolving the image with a Gaussian function, 
          which effectively averages pixel values in a neighborhood, 
          giving higher weight to pixels closer to the center of the kernel. 
        """
        
        # The size of the kernel determines how many neighboring pixels are considered when computing the smoothed value for a single pixel.
        # sigmaX=0 means that the function will automatically calculate sigmaX based on the kernel size. In other words, OpenCV will choose an appropriate value for sigmaX depending on the kernel size.
        flat_field = cv2.GaussianBlur(float_image, (self.kernel_size, self.kernel_size), sigmaX=0)
        
        # Normalize the illumination pattern
        """
        This step effectively rescales the illumination pattern, 
        making it independent of the global lighting conditions 
        and ensuring it is comparable across different images or regions within the image.
        """
        mean_illumination = np.mean(flat_field)
        flat_field_normalized = flat_field / mean_illumination

        return flat_field_normalized
    
    def dark_frame_sub(self,image):
        # Convert to float32 for precise calculations
        image_float = image.astype(np.float32)
        
        # Estimate and remove dark current
        dark_corrected = image_float - self.dark_frame
        dark_corrected = np.clip(dark_corrected, 0, None)

        return dark_corrected
    


    # function to update access token for sentinel hub api
    def update_access_token(self):
        CLIENT_ID = "ff2b6aaf-b452-425a-ab9a-92472076a89f"
        CLIENT_SECRET = "zfaPUlXe58UiUnzG5uuOzk4KGE9kDkJN"

        url = "https://services.sentinel-hub.com/oauth/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            token = response.json().get("access_token")
            # print("New Access Token:", token)  # Print the token
            return token
        else:
            print("Failed to get new token:", response.text)
            return None
            
    # function to give me image bounds
    def get_image_bounds(self,image_path):
        with rasterio.open(image_path) as src:
            bbox = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return bbox

    # function to get metadata        
    def get_sentinel_metadata(self, bbox, date):
        token = self.update_access_token()
        if token is None:
            print("No valid token, using default metadata values.")
            return 45.0, 0.1  # default values

        headers = {"Authorization": f"Bearer {token}"}
        
        payload = {
            "input": {
                "bounds": {
                    "bbox": bbox,
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "S2L2A",
                    "dataFilter": {"timeRange": {"from": date, "to": date}}
                }]
            },
            "output": {
                "responses": [
                    {
                        "identifier": "default",
                        "format": {
                            "type": "application/json",
                            "sampleType": "FLOAT32"
                        }
                    }
                ]
            },
            "evalscript": """
            function setup() {
                return {
                    input: ["sunZenithAngles", "AOT"],
                    output: { bands: 2, sampleType: "FLOAT32" }
                };
            }
            function evaluatePixel(sample) {
                return [sample.sunZenithAngles, sample.AOT];
            }
            """
        }
        
        # making the request to sentinel hub 
        response = requests.post("https://services.sentinel-hub.com/api/v1/process", json=payload, headers=headers)
        
        try:
            data = response.json()
        except Exception as e:
            print("Error decoding JSON:", e, "Response text:", response.text)
            print("Using default metadata values.")
            return 45.0, 0.1

        if "default" not in data or not data["default"]:
            print("No 'default' output found in the response. Response:", data)
            print("Using default metadata values.")
            return 45.0, 0.1

        try:
            solar_z = np.mean(data["default"][0])
            aot = np.mean(data["default"][1]) if data["default"][1] is not None else 0.1
        except Exception as e:
            print("Error processing metadata:", e, "Data:", data)
            print("Using default metadata values.")
            return 45.0, 0.1

        return solar_z, aot


    # gives the aeroprofrofile based on the aot value (aerosol, optical thickness)
    def get_aero_profile(self,aot):
        if aot < 0.05:
            return AeroProfile.PredefinedType(AeroProfile.Maritime)
        elif aot < 0.2:
            return AeroProfile.PredefinedType(AeroProfile.Continental)
        else:
            return AeroProfile.PredefinedType(AeroProfile.Urban)
        
    # function to convert toa data to sr for one band of the image
    '''
    Converts Top-of-Atmosphere (TOA) reflectance to Surface Reflectance (SR)
    using the 6S (Second Simulation of the Satellite Signal in the Solar Spectrum) model
    '''
    def TOA_to_SR(self,toa_image,wavelength,solar_z,aero_profile):
        
        # Create an instance of the 6S model
        s = SixS()

        # sets the atmospheric profile to a predefined Midlatitude model
        # This defines temperature, pressure, humidity, and gas concentration values.
        s.atmos_profile = AtmosProfile.PredefinedType(AtmosProfile.MidlatitudeSummer)

        # the aero profile affects how light scatters in the atmosphere
        s.aero_profile = aero_profile

        # Define the ground reflectance as a homogeneous Lambertian surface with a reflectance of 0.3
        s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.3) 

        # the angle between the sun and the vertical direction, shows how muchh sunlight actually reaches the surface
        s.geometry.solar_z = solar_z 
        
        # defines the wavelength for which the odel has to run
        s.wavelength = Wavelength(wavelength)

        # runs the 6S model simulation to compute atmospheric parameters.
        s.run()


        # Extract path radiance, which represents the portion of reflectance caused by atmospheric scattering, some light which is scattered reaches sensor directly before even reaching the ground
        # P: Atmospheric intrinsic reflectance (path radiance contribution)
        P = s.outputs.atmospheric_intrinsic_reflectance

        # represents how much light reaches the ground after passing through the atmosphere, as some light is lost as sunlight passes through the atmosphere
        # T_down: Downward atmospheric transmittance
        T_down = s.outputs.transmittance_total_scattering.downward

        # accounts for the light that is scattered back to the surface
        # S: Spherical albedo of the atmosphere
        S = s.outputs.spherical_albedo

        # Correct TOA reflectance to Surface Reflectance (SR)
        sr_image = (toa_image - P) / (T_down - toa_image * S)


        return sr_image


    # combines the metadata functions, aeroprofile function and applies toa_to_sr function on all bands of the image individually
    def atmospheric_correction(self,tif_image,bbox,date):
        solar_z, aot = self.get_sentinel_metadata(bbox, date)

        aero_profile = self.get_aero_profile(aot)

       
        toa_image = tif_image.astype(np.float32) * self.scalefactor

        wavelengths = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 0.865, 0.945, 1.375, 1.610, 2.190]
        if toa_image.shape[0] != len(wavelengths):
            print("Warning: Number of bands and wavelengths do not match.")
            wavelengths = wavelengths[:toa_image.shape[0]]


        sr_bands = np.zeros_like(toa_image, dtype=np.float32)

        for i in range(toa_image.shape[0]):
            print(f"Processing Band {i+1} at {wavelengths[i]} nm")
            sr_bands[i] = self.TOA_to_SR(toa_image[i], wavelengths[i], solar_z, aero_profile)

        return sr_bands
    
    # a combined function which calls all corrections at once, if we need all corrections
    def apply_corrections(self,image,bbox=None,date=None):
        """
        I_observed(x,y) = I_true(x,y) * F(x,y) + D(x,y)
        where:
        - I_observed is the captured image
        - I_true is the actual scene radiance
        - F(x,y) is the flat field pattern
        - D(x,y) is the dark current pattern
        """

        """
        Applies the complete flat field correction process.
        The correction follows the formula:
        I_corrected = (I_observed - D) / F 
        """
        # Subtract the dark current
        float_image = image.astype(np.float32)
        processed_image = self.dark_frame_sub(float_image)

        # Estimate flat field pattern
        flat_field = self.estimate_flat_field(image)

        # Apply flat field correction
        # Add small epsilon to avoid division by zero
        epsilon = np.finfo(np.float32).eps


        """
        Why division with the flat-field pattern F?
        Dividing by F helps correct for the uneven distribution of light across the image. 
        Since the flat field pattern represents how much more or less light each pixel received, 
        dividing by this pattern ensures that the corrected image reflects a uniform illumination, 
        where variations due to sensor imperfections or environmental factors are removed.
        """
        flatfield_and_darkframe_corrected_image = processed_image / (flat_field + epsilon)

        # final_image = self.atmospheric_correction(flatfield_and_darkframe_corrected_image, bbox, date)
        return flatfield_and_darkframe_corrected_image
    

class GeometricCorrection:
    def __init__(self):
        pass

    def update_access_token(self):
        CLIENT_ID = "ff2b6aaf-b452-425a-ab9a-92472076a89f"
        CLIENT_SECRET = "zfaPUlXe58UiUnzG5uuOzk4KGE9kDkJN"

        url = "https://services.sentinel-hub.com/oauth/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            token = response.json().get("access_token")
            # print("New Access Token:", token)  # Print the token
            return token
        else:
            print("Failed to get new token:", response.text)
            return None


    def get_image_bounds(self,image_path):
        with rasterio.open(image_path) as src:
            bbox = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return bbox
        

    def dem_request_sentinel_hub(self,bbox):

        access_token = self.update_access_token()
        url = "https://services.sentinel-hub.com/api/v1/process"
        headers = {"Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"}
        
        payload = {
            "input": {
                "bounds": {
                    "bbox": list(bbox),
                    "properties": { "crs": "http://www.opengis.net/def/crs/EPSG/0/4326" }
                },
                "data": [{
                    "type": "DEM",
                    "dataFilter": {
                        "demInstance": "COPERNICUS_90"
                    }
                }]
            },
            "output": {
                "width": 256,
                "height": 256,
                "responses": [{
                    "identifier": "default",
                    "format": { "type": "image/tiff" }
                }]
            },
            "evalscript": """
            function setup() {
                return {
                    input: ["DEM"],
                    output: { bands: 1 }
                };
            }
            
            function evaluatePixel(sample) {
                return [sample.DEM];
            }
            """
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            with open("dem_output1.tif", "wb") as f:
                f.write(response.content)
            print("DEM data saved as dem_output1.tif")
        else:
            print("Error:", response.text)


# class ocntaining all relevant functionalities related to geometric corrections
class GeometricCorrection:
    def __init__(self):
        pass
    
    def update_access_token(self):
        CLIENT_ID = "ff2b6aaf-b452-425a-ab9a-92472076a89f"
        CLIENT_SECRET = "zfaPUlXe58UiUnzG5uuOzk4KGE9kDkJN"
        url = "https://services.sentinel-hub.com/oauth/token"
        payload = {
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET
        }
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            token = response.json().get("access_token")
            return token
        else:
            print("Failed to get new token:", response.text)
            return None

    def get_image_bounds(self, image_path):
        with rasterio.open(image_path) as src:
            bbox = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
        return bbox
        

    def dem_request_sentinel_hub(self, bbox):
        access_token = self.update_access_token()
        url = "https://services.sentinel-hub.com/api/v1/process"
        headers = {"Authorization": f"Bearer {access_token}",
                   "Content-Type": "application/json"}
        payload = {
            "input": {
                "bounds": {
                    "bbox": list(bbox),
                    "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                },
                "data": [{
                    "type": "DEM",
                    "dataFilter": {"demInstance": "COPERNICUS_90"}
                }]
            },
            "output": {
                "width": 256,
                "height": 256,
                "responses": [{
                    "identifier": "default",
                    "format": {"type": "image/tiff"}
                }]
            },
            "evalscript": """
            function setup() {
                return {
                    input: ["DEM"],
                    output: { bands: 1 }
                };
            }
            function evaluatePixel(sample) {
                return [sample.DEM];
            }
            """
        }
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            with open("dem_output1.tif", "wb") as f:
                f.write(response.content)
            print("DEM data saved as dem_output1.tif")
            return "dem_output1.tif"
        else:
            print("Error:", response.text)
            return None


    # function to correct terrain distortions using orthorectification
    # uses dem data (digital elevation model) and rpcs ( ational polynomial coefficients)    
    def terrain_distortion_corr(self, input_og, input_dem, output_image):
        # .warp() is a function from the gdal library which transforms and reprojects images correctly
        # output_image is the destination fole where the image is going to be stored
        # input_og is the original image which requires corrction
        # rpc = true, just enables the use of rational polynomial coefficients, these are functions which map the image coordinates to real world geographic coordinates
        # options parameter uses the provided dem data to correct for distortions
        gdal.Warp(
            output_image,
            input_og,
            rpc=True, 
            options=f"-to RPC_DEM={input_dem}"  
        )
        print(f"Orthorectified image saved to: {output_image}")
        return output_image


class masking:
    def __init__(self, bands):
        self.red = bands['red']
        self.green = bands['green']
        self.blue = bands['blue']
        self.nir = bands['nir']
        self.swir1 = bands['swir1'] 
    
    def calculate_indices(self):
        self.ndvi = (self.nir - self.red) / (self.nir + self.red)
        self.ndwi = (self.nir - self.swir1) / (self.nir + self.swir1)
        self.savi = (1.5 * (self.nir - self.red)) / (self.nir + self.red + 0.5)
        self.mndwi = (self.green - self.swir1) / (self.green + self.swir1)
        self.ndbi = (self.swir1 - self.nir) / (self.swir1 + self.nir)
    
    def generate_masks(self):
        self.vegetation_mask = self.ndvi > 0.3
        self.water_mask = (self.ndwi > 0.3) | (self.mndwi > 0.3) 
        self.soil_mask = (self.savi > 0.2) & (self.savi < 0.5)
        self.urban_mask = self.ndbi > 0
    
    def producee_mask(self):
        self.calculate_indices()
        self.generate_masks()
        return {
            'vegetation_mask': self.vegetation_mask,
            'water_mask': self.water_mask,
            'soil_mask': self.soil_mask,
            'urban_mask': self.urban_mask
        }



# creates instances of the radiometric and geometric classes   
rc = RadiometricCorrection()
gc = GeometricCorrection()

# difines an estimate date of image acquisition (used in rc)
date = "2024-03-01T00:00:00Z"

#file directories used for processing
input_folder = Path("/home/goodarth/Desktop/anant/payload/test_input")
output_folder_rad = Path("/home/goodarth/Desktop/anant/payload/output_radiometric")
output_folder_geo = Path("/home/goodarth/Desktop/anant/payload/output_geometric")


# for loop running across all files in the input folder
for tif_image in input_folder.glob("*.tif"):
    
    print(f"Processing: {tif_image.name}")


    with rasterio.open(str(tif_image)) as src:
        image_data = src.read().astype(np.float32)
        # gets image's metadata and ensures that data output is in float32
        profile = src.profile
        profile.update(dtype=rasterio.float32)

        # defines all relevant bands for masking purposes in a dictionary
        bands = {
            'red': src.read(4).astype(np.float32), 
            'green': src.read(3).astype(np.float32),  
            'blue': src.read(2).astype(np.float32),  
            'nir': src.read(8).astype(np.float32),  
            'swir1': src.read(11).astype(np.float32) 
        }
    # function to get bounds
    bbox = rc.get_image_bounds(str(tif_image))
    rc_corrected_image = rc.apply_corrections(image_data,bbox,date)
    
    # sets path for radiometically corrected image
    rad_output_path = output_folder_rad / f"{tif_image.stem}_rad.tif"
    
    # writes the corrected image into the path mentioned above
    with rasterio.open(str(rad_output_path), "w", **profile) as dst:
        dst.write(rc_corrected_image)
        print(f"Radiometric correction saved: {rad_output_path}")

    # gets dem data for the particular image
    dem_file = gc.dem_request_sentinel_hub(bbox)

    # sets path for the geometrically corrected image
    geo_output_path = output_folder_geo / f"{tif_image.stem}_geo.tif"

    # applies geometric corrections and stores it in thr output_folder_geo folder 
    gc.terrain_distortion_corr(str(rad_output_path), dem_file, str(geo_output_path))
    print(f"Final corrected image saved: {geo_output_path}")


    # creates an instance of the masking class, which takes input as a dictionary of all bands and then applies the produce mask function on it to get the masks as a dictionary
    masking_timee = masking(bands)
    masks = masking_timee.producee_mask()

