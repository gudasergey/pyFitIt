Pyfitit sample consists of
- parameters table (params.csv) where each row corresponds to one sample object
- spectrum tables of different spectrum types (spectrumType_spectra.csv files), the first column contains energy/wavelength values, other columns are spectra (number of columns in spectrumType_spectra.csv files equals to the number of rows in params.csv)
- meta information in JSON format:
  * nameColumn - name of the parameter to use as index in the sample (names should be short to be pretty displayed on scatter plots!)
  * labels - list of parameter names to predict
  * features - list of parameter names to use as sample object features
  * labelMaps - dict of dicts with label encoding in the format "stringLabelValue":number
- sample in binary format (binary_repr.pkl), it is used for round-trip save/load floats (text format is not round-trip and modify floats)