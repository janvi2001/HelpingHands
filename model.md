# MODEL TRAINING

## Imports


```python
# To use the wavelet decomposition function in the PyWavelets module
import pywt  

# For data handling
import numpy as np
import pandas as pd

# To handle file path
from glob import glob

# To use mathematical functions
from math import sqrt,log10

# To save and load our model 
import joblib

# To train a decision tree classification model
from sklearn.tree import DecisionTreeClassifier

# To ignore any warnings that may be prompted
import warnings
warnings.filterwarnings('ignore')
```

## Denoising EEG Signals using Discrete Wavelet Transform 


```python
def madev(d, axis=None):
    '''
    Median absolute deviation of a signal
    ''' 
    return np.median(np.absolute(d))


def wavelet_denoising(x):
    ''' 
    Function to denoise the EEG signals using the discrete wavelet transform. 
    '''
    # Using wavelet decomposition to apply DWT on 4 levels
    c = pywt.wavedec(x,"sym18", mode="per",level = 4)
    
    # Calculation for universal threshold
    sigma = (1/0.6745) * madev(c[-1])
    univ_thresh = sigma * np.sqrt(2 * np.log(len(x)))
    
    # Applying hard thresholding using the universal threshold 
    # calculated in the previous step
    c[1:] = (pywt.threshold(i, value=univ_thresh, mode='hard') for i in c[1:])
    
    return pywt.waverec(c, "sym18", mode='per')
```

## Feature Extraction of EEG Signals using Discrete Wavelet Transform


```python
def FEdwt(s):
    '''
    Function to extract features (namely information relating to the alpha(8-12Hz) and 
    beta(13-30Hz) frequency bands that corespond to the performance of motor functions.)
    from EEG signals.
    '''
    # Using wavelet decomposition to apply DWT on 5 levels 
    # to get frequencies upto approx. the alpha band 
    coefli = pywt.wavedec(s,"sym18", mode="per", level=5)
    
    # Making a list of features and appending the original signal
    features = []
    features.append(s)
    
    # Appending only the decompositions coresponding to the alpha and beta bands
    for c in coefli[1:3:]:
        features.append(pd.DataFrame(pywt.idwt(None, c,"sym18", mode="per")))
    
    # Final dataframe of signal and features extracted from it
    featr = pd.concat([feature for feature in features])
    return featr
```

## Data PreProcessing 

### Train - Test split : 75% - 25%


```python
# Get the file path to all files present in the train and test folders
train_fnames =  glob('data/train/subj*_data.csv')
test_fnames =  glob('data/test/subj*_data.csv')
```


```python
def preprocess_data(fname):
    
    # Store data in a dataframe and drop the id column
    datax = pd.read_csv(fname)
    datax.drop(['id'], axis = 1, inplace=True)
    
    # Denoise the signal and re-attach the header to the result
    cols = datax.columns

    datax = pd.DataFrame(wavelet_denoising(datax))
    datax.columns = cols
    
    # Get output filename from data filename
    events = fname.replace('_data','_events') 
    datay = pd.read_csv(events)

    # Drop the id column
    datay.drop(['id'], axis = 1, inplace=True)
    
    # Channel Selection to reduce dimensionality and preserve raw data
    # Drop all channels that are away from the central lobe
    datax.drop([x for x in datax.columns if 'C' not in x], axis = 1, inplace=True)

    # Concatenating all labels to end of the data
    for col in datay.columns:
        datax[f'{col}_output'] = datay[col]
    
    return datax
```


```python
# Concatenating all training data 
train_data = pd.concat([preprocess_data(fname) for fname in train_fnames])
train_data

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FC5</th>
      <th>FC1</th>
      <th>FC2</th>
      <th>FC6</th>
      <th>C3</th>
      <th>Cz</th>
      <th>C4</th>
      <th>CP5</th>
      <th>CP1</th>
      <th>CP2</th>
      <th>CP6</th>
      <th>HandStart_output</th>
      <th>FirstDigitTouch_output</th>
      <th>BothStartLoadPhase_output</th>
      <th>LiftOff_output</th>
      <th>Replace_output</th>
      <th>BothReleased_output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>292.183972</td>
      <td>300.131051</td>
      <td>308.161367</td>
      <td>315.968030</td>
      <td>329.734439</td>
      <td>335.162066</td>
      <td>339.322013</td>
      <td>342.841195</td>
      <td>340.881812</td>
      <td>337.439870</td>
      <td>332.651406</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>288.927679</td>
      <td>296.780747</td>
      <td>304.716068</td>
      <td>312.430381</td>
      <td>326.033940</td>
      <td>331.397360</td>
      <td>335.508097</td>
      <td>338.985647</td>
      <td>337.049443</td>
      <td>333.648218</td>
      <td>328.916400</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>271.703353</td>
      <td>279.387597</td>
      <td>287.152326</td>
      <td>294.700799</td>
      <td>308.011911</td>
      <td>313.260029</td>
      <td>317.282394</td>
      <td>320.685185</td>
      <td>318.790605</td>
      <td>315.462499</td>
      <td>310.832405</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>249.053309</td>
      <td>260.469669</td>
      <td>272.005604</td>
      <td>283.220250</td>
      <td>302.996359</td>
      <td>310.793405</td>
      <td>316.769369</td>
      <td>321.824842</td>
      <td>319.010095</td>
      <td>314.065580</td>
      <td>307.186722</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>249.304860</td>
      <td>265.001491</td>
      <td>280.862528</td>
      <td>296.281817</td>
      <td>323.472469</td>
      <td>334.192817</td>
      <td>342.409316</td>
      <td>349.360209</td>
      <td>345.490144</td>
      <td>338.691810</td>
      <td>329.233902</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>141421</th>
      <td>81.553617</td>
      <td>55.054955</td>
      <td>28.278747</td>
      <td>2.248287</td>
      <td>-43.654296</td>
      <td>-61.752120</td>
      <td>-75.623009</td>
      <td>-87.357333</td>
      <td>-80.823986</td>
      <td>-69.347209</td>
      <td>-53.380602</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>141422</th>
      <td>107.748201</td>
      <td>81.887391</td>
      <td>55.755716</td>
      <td>30.351838</td>
      <td>-14.445819</td>
      <td>-32.108009</td>
      <td>-45.645010</td>
      <td>-57.096876</td>
      <td>-50.720793</td>
      <td>-39.520275</td>
      <td>-23.938002</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>141423</th>
      <td>120.161328</td>
      <td>96.299704</td>
      <td>72.188155</td>
      <td>48.748139</td>
      <td>7.413594</td>
      <td>-8.883209</td>
      <td>-21.373723</td>
      <td>-31.940296</td>
      <td>-26.057120</td>
      <td>-15.722465</td>
      <td>-1.344790</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>141424</th>
      <td>130.386958</td>
      <td>104.762738</td>
      <td>78.870133</td>
      <td>53.698665</td>
      <td>9.310845</td>
      <td>-8.189760</td>
      <td>-21.602916</td>
      <td>-32.950013</td>
      <td>-26.632263</td>
      <td>-15.534214</td>
      <td>-0.094498</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>141425</th>
      <td>149.150528</td>
      <td>125.297457</td>
      <td>101.194552</td>
      <td>77.762938</td>
      <td>36.443210</td>
      <td>20.152249</td>
      <td>7.666212</td>
      <td>-2.896573</td>
      <td>2.984494</td>
      <td>13.315444</td>
      <td>27.687966</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>1807279 rows × 17 columns</p>
</div>




```python
# Concatenating all testing data 
test_data = pd.concat([preprocess_data(fname) for fname in test_fnames])
test_data

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FC5</th>
      <th>FC1</th>
      <th>FC2</th>
      <th>FC6</th>
      <th>C3</th>
      <th>Cz</th>
      <th>C4</th>
      <th>CP5</th>
      <th>CP1</th>
      <th>CP2</th>
      <th>CP6</th>
      <th>HandStart_output</th>
      <th>FirstDigitTouch_output</th>
      <th>BothStartLoadPhase_output</th>
      <th>LiftOff_output</th>
      <th>Replace_output</th>
      <th>BothReleased_output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-373.220094</td>
      <td>-380.495103</td>
      <td>-387.846311</td>
      <td>-394.992779</td>
      <td>-407.594989</td>
      <td>-412.563611</td>
      <td>-416.371760</td>
      <td>-419.593331</td>
      <td>-417.799649</td>
      <td>-414.648786</td>
      <td>-410.265274</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-376.130104</td>
      <td>-383.550627</td>
      <td>-391.048872</td>
      <td>-398.338283</td>
      <td>-411.192561</td>
      <td>-416.260565</td>
      <td>-420.144883</td>
      <td>-423.430891</td>
      <td>-421.601333</td>
      <td>-418.387447</td>
      <td>-413.916256</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-386.955898</td>
      <td>-393.351202</td>
      <td>-399.813490</td>
      <td>-406.095796</td>
      <td>-417.174128</td>
      <td>-421.541937</td>
      <td>-424.889598</td>
      <td>-427.721612</td>
      <td>-426.144825</td>
      <td>-423.374969</td>
      <td>-419.521518</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-397.889565</td>
      <td>-403.693940</td>
      <td>-409.559108</td>
      <td>-415.260926</td>
      <td>-425.315613</td>
      <td>-429.279833</td>
      <td>-432.318169</td>
      <td>-434.888503</td>
      <td>-433.457412</td>
      <td>-430.943492</td>
      <td>-427.446102</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-379.715488</td>
      <td>-385.088507</td>
      <td>-390.517802</td>
      <td>-395.795886</td>
      <td>-405.103353</td>
      <td>-408.772971</td>
      <td>-411.585510</td>
      <td>-413.964828</td>
      <td>-412.640090</td>
      <td>-410.312994</td>
      <td>-407.075514</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>151118</th>
      <td>123.425156</td>
      <td>135.066750</td>
      <td>146.830278</td>
      <td>158.266179</td>
      <td>178.432451</td>
      <td>186.383325</td>
      <td>192.477190</td>
      <td>197.632402</td>
      <td>194.762122</td>
      <td>189.720057</td>
      <td>182.705486</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>151119</th>
      <td>138.938076</td>
      <td>144.035821</td>
      <td>149.186960</td>
      <td>154.194634</td>
      <td>163.025256</td>
      <td>166.506870</td>
      <td>169.175316</td>
      <td>171.432735</td>
      <td>170.175867</td>
      <td>167.967993</td>
      <td>164.896378</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>151120</th>
      <td>137.225714</td>
      <td>138.581369</td>
      <td>139.951223</td>
      <td>141.282926</td>
      <td>143.631274</td>
      <td>144.557147</td>
      <td>145.266773</td>
      <td>145.867094</td>
      <td>145.532852</td>
      <td>144.945707</td>
      <td>144.128866</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>151121</th>
      <td>124.134817</td>
      <td>124.646762</td>
      <td>125.164069</td>
      <td>125.666968</td>
      <td>126.553790</td>
      <td>126.903434</td>
      <td>127.171415</td>
      <td>127.398118</td>
      <td>127.271896</td>
      <td>127.050169</td>
      <td>126.741699</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>151122</th>
      <td>116.835845</td>
      <td>113.426805</td>
      <td>109.982060</td>
      <td>106.633255</td>
      <td>100.727911</td>
      <td>98.399635</td>
      <td>96.615153</td>
      <td>95.105538</td>
      <td>95.946050</td>
      <td>97.422531</td>
      <td>99.476627</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>636910 rows × 17 columns</p>
</div>



## Training the Model on Train Data

AIM :: To make multi-label prediction 

To acheive multi-label prediction we train a binary classification model on our data seperately for each label and concatenate the predictions while giving the output.

### Training Decision Tree Classifier on the label HandStart


```python
clf1 = DecisionTreeClassifier()
clf1.fit(train_data.iloc[:,:11] , train_data.iloc[:,11])

```




    DecisionTreeClassifier()



### Training Decision Tree Classifier on the label FirstDigitTouch


```python
clf2 = DecisionTreeClassifier()
clf2.fit(train_data.iloc[:,:11] , train_data.iloc[:,12])

```




    DecisionTreeClassifier()



### Training Decision Tree Classifier on the label BothStartLoadPhase


```python
clf3 = DecisionTreeClassifier()
clf3.fit(train_data.iloc[:,:11] , train_data.iloc[:,13])

```




    DecisionTreeClassifier()



### Training Decision Tree Classifier on the label LiftOff


```python
clf4 = DecisionTreeClassifier()
clf4.fit(train_data.iloc[:,:11] , train_data.iloc[:,14])

```




    DecisionTreeClassifier()



### Training Decision Tree Classifier on the label Replace


```python
clf5 = DecisionTreeClassifier()
clf5.fit(train_data.iloc[:,:11] , train_data.iloc[:,15])

```




    DecisionTreeClassifier()



### Training Decision Tree Classifier on the label BothReleased


```python
clf6 = DecisionTreeClassifier()
clf6.fit(train_data.iloc[:,:11] , train_data.iloc[:,16])

```




    DecisionTreeClassifier()



## Accuracy Scores for each Label

WARNING :: The high accuracy only applies to each label seperately


```python
score1 = clf1.score(test_data.iloc[:,:11], test_data.iloc[:,11]) * 100
print("HandStart Acc          :: %.2f" % score1,"%")

score2 = clf2.score(test_data.iloc[:,:11], test_data.iloc[:,12]) * 100
print("FirstDigitTouch Acc    :: %.2f" % score2,"%")

score3 = clf3.score(test_data.iloc[:,:11], test_data.iloc[:,13]) * 100
print("BothStartLoadPhase Acc :: %.2f" % score3,"%")

score4 = clf4.score(test_data.iloc[:,:11], test_data.iloc[:,14]) * 100
print("LiftOff Acc            :: %.2f" % score4,"%")


score5 = clf5.score(test_data.iloc[:,:11], test_data.iloc[:,15]) * 100
print("Replace Acc            :: %.2f" % score5,"%")


score6 = clf6.score(test_data.iloc[:,:11], test_data.iloc[:,16]) * 100
print("BothReleased Acc       :: %.2f" % score6,"%")

```

    HandStart Acc          :: 95.53 %
    FirstDigitTouch Acc    :: 95.36 %
    BothStartLoadPhase Acc :: 95.36 %
    LiftOff Acc            :: 95.23 %
    Replace Acc            :: 95.01 %
    BothReleased Acc       :: 95.10 %
    

## Saving models to pickle files using Joblib


```python
joblib.dump(clf1, 'pickle1.pkl')
joblib.dump(clf2, 'pickle2.pkl')
joblib.dump(clf3, 'pickle3.pkl')
joblib.dump(clf4, 'pickle4.pkl')
joblib.dump(clf5, 'pickle5.pkl')
joblib.dump(clf6, 'pickle6.pkl')
```




    ['pickle6.pkl']


