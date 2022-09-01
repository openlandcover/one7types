import ee
import math
import configparser as cp
import io
import ast
import subprocess

from areaOfInterestMask import semiarid

def filterAndMaskCloudsL8C2SR(config):
    # Mask clouds and shadows
    def maskAndPrepSrL8(image):
        # Apply the scaling factors to the appropriate bands.
        # Based on discussion forum post
        # https://groups.google.com/g/google-earth-engine-developers/c/nvz0PP6P5II/m/gZnZzZQgAwAJ
        def getFactorImg(factorNames):
            factorList = image.toDictionary().select(factorNames).values()
            return ee.Image.constant(factorList)

        # Develop masks for unwanted pixels (fill, cloud, cloud shadow).
        qaMask = image.select("QA_PIXEL").bitwiseAnd(int("11111", 2)).eq(0)
        saturationMask = image.select("QA_RADSAT").eq(0)

        scaleImg = getFactorImg(["REFLECTANCE_MULT_BAND_.|TEMPERATURE_MULT_BAND_ST_B10"])
        offsetImg = getFactorImg(["REFLECTANCE_ADD_BAND_.|TEMPERATURE_ADD_BAND_ST_B10"])
        scaled = image.select("SR_B.|ST_B10").multiply(scaleImg).add(offsetImg)

        # Replace original bands with scaled bands and apply masks.
        return image.addBands(scaled, None, True) \
            .updateMask(qaMask).updateMask(saturationMask) \
            .copyProperties(image, ["system:time_start"])

    allIndia = ee.FeatureCollection(config.get('CORE', 'indiaMainlandID'))
    configSeas = config['FEATURES-SEASONALITY']
    l8c2t1SR = ee.ImageCollection(configSeas.get('landsat8C2T1SREEID'))
    l8compStDt  = configSeas.get('landsat8CompositingStartDate')
    l8compEndDt = configSeas.get('landsat8CompositingEndDate')
    cloudcoverMinThresh = configSeas.getint('minCloudcoverPerc')

    india = ee.Feature(allIndia.first()).simplify(5e3)
    maskedColl = l8c2t1SR \
        .filterDate(l8compStDt, l8compEndDt) \
        .filterMetadata('CLOUD_COVER', 'less_than', cloudcoverMinThresh) \
        .filterBounds(india.geometry()) \
        .map(maskAndPrepSrL8)

    return maskedColl

def seasonalityParamsL8(returnExisting = False, startFreshExport = False):
    # Add (indepependent) vars for harmonic fit
    def addVariables(img):
        date = ee.Date(img.get('system:time_start'))
        years = date.difference(ee.Date('1970-01-01'), 'year')
        img = img.addBands(img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI').float()) \
                .addBands(ee.Image(years).rename('t').float()) \
                .addBands(ee.Image.constant(1))
        timeRadians = img.select('t').multiply(2 * math.pi)
        return img.addBands(timeRadians.cos().rename('cos')) \
                .addBands(timeRadians.sin().rename('sin'))

    # Use harmonic fit parms to calculate fitted NDVI
    def fittedNDVI(img):
        return img.addBands(
            img.select(harmonicIndependents) \
                .multiply(harmonicTrendCoefficients) \
                .reduce('sum') \
                .rename('fitted'))

    # Load the configuration file and read-in parameters
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configSeas = config['FEATURES-SEASONALITY']
    phaseBandName  = configSeas.get('featureBandNamePhase')
    amplBandName   = configSeas.get('featureBandNameAmplitude')
    offsetBandName = configSeas.get('featureBandNameOffset')
    amplSegBandName  = configSeas.get('featureBandNameAmplitudeSegmented')
    phaseSegBandName = configSeas.get('featureBandNamePhaseSegmented')
    trendBandName  = configSeas.get('featureBandNameTrend')
    configAssemble = config['FEATURES-ASSEMBLE']
    snicRunComp       = configAssemble.getint('snicCompactness')
    snicRunNeighbSize = configAssemble.getint('snicNeighbourhoodSize')
    snicSeedGridFine  = configAssemble.getint('snicSuperPixSeedLocSpacingFINE')
    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    districts = ee.FeatureCollection(config.get('CORE', 'indiaDistricts'))
    trialSpanOfExportDistrictName = config.get('CORE', 'trialSpanToGenerateFeatureRastersOverDistrict')
    trialSpanOfExport = ee.Feature(districts.filter(ee.Filter.eq('DISTRICT', trialSpanOfExportDistrictName)).first())
    fullSpanOfExport  = ee.Feature(ee.FeatureCollection(config.get('CORE', 'fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)

    fine = ee.Algorithms.Image.Segmentation.seedGrid(snicSeedGridFine)
    snicRunParams = {"seeds": fine, "compactness": snicRunComp, "neighborhoodSize": snicRunNeighbSize}

    if returnExisting == True:
        assetName = configSeas.get('existingSeasonality')
        LS8srHarmonicModel = ee.Image(assetFolder + assetName)
    else:
        # Filter to time period of interest, mask clouds
        filteredLS8sr = filterAndMaskCloudsL8C2SR(config)

        # Fit harmonic model with trend
        # Define independent and dependent vars
        dependent = ee.String('NDVI')
        harmonicIndependents = ee.List(['constant', 't', 'cos', 'sin'])
        # Assemble those vars and perform harmonic fit
        harmonicLS8sr = filteredLS8sr.map(addVariables)
        harmonicTrend = harmonicLS8sr.select(harmonicIndependents.add(dependent))\
            .reduce(ee.Reducer.linearRegression(harmonicIndependents.length(), 1))
        harmonicTrendCoefficients = harmonicTrend.select('coefficients')\
            .arrayProject([0])\
            .arrayFlatten([harmonicIndependents])

        # Extract seasonality params
        fittedHarmonic = harmonicLS8sr.map(fittedNDVI)
        phase = harmonicTrendCoefficients.select('cos').atan2(
            harmonicTrendCoefficients.select('sin'))
        amplitude = harmonicTrendCoefficients.select('cos').hypot(
            harmonicTrendCoefficients.select('sin'))

        # Assemble seasonality bands of interest
        LS8srHarmonicModel = ee.Image.cat(phase, amplitude, harmonicTrendCoefficients.select(['constant', 't']))\
            .rename([phaseBandName, amplBandName, offsetBandName, trendBandName]).float()

        harmonicModel_seg = ee.Algorithms.Image.Segmentation.SNIC(** {"image": LS8srHarmonicModel.select([phaseBandName, amplBandName])} | snicRunParams).select([phaseBandName + "_mean", amplBandName + "_mean"]).rename([phaseSegBandName, amplSegBandName])
        phase_seg = harmonicModel_seg.select(phaseSegBandName)
        ampl_seg  = harmonicModel_seg.select(amplSegBandName)

        LS8srHarmonicModel = LS8srHarmonicModel.addBands(ee.Image.cat(phase_seg, ampl_seg)) \
            .set(dict(
                lt8eeid = configSeas.get('landsat8C2T1SREEID'),
                timeseriesFrom = configSeas.get('landsat8CompositingStartDate'),
                timeseriesTo   = configSeas.get('landsat8CompositingEndDate'),
                snicSegmSeedGridSpacing = snicSeedGridFine,
                snicSegmNeighbSize = snicRunNeighbSize,
                snicCompact = snicRunComp))

        if startFreshExport == True:
            reg = fullSpanOfExport
            # reg = trialSpanOfExport
            ee.batch.Export.image.toAsset(** {
              'image': LS8srHarmonicModel,
              'description': 'landsat8Phenology',
              'assetId': assetFolder + 'lt8Phenology',
              'scale': 30,
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return LS8srHarmonicModel

def tasselledCapCoeffsL8(returnExisting = False, startFreshExport = False):
    def filterAndMaskCloudsL8C1SR(config):
        # Mask clouds and shadows
        def maskLS8C1SR(image):
            # Bits 3 and 5 are cloud shadow and cloud, respectively.
            cloudShadowBitMask = 1 << 3
            cloudsBitMask = 1 << 5

            # Get the pixel QA band.
            qa = image.select('pixel_qa')

            # Both flags should be set to zero, indicating clear conditions.
            mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) \
                .And(qa.bitwiseAnd(cloudsBitMask).eq(0))

            # Return the masked image, scaled to reflectance, without the QA bands.
            return image.updateMask(mask).divide(10000) \
                .select("B[0-9]*") \
                .copyProperties(image, ["system:time_start"])

        allIndia = ee.FeatureCollection(config.get('CORE', 'indiaMainlandID'))
        configTct = config['FEATURES-TASSELLEDCAP']
        l8c1t1SR = ee.ImageCollection(configTct.get('landsat8C1T1SREEID'))
        l8compStDt  = configTct.get('landsat8TCCalc3yrsStartDate')
        l8compEndDt = configTct.get('landsat8TCCalc3yrsEndDate')
        cloudcoverMinThresh = configTct.getint('minCloudcoverPerc')

        india = ee.Feature(allIndia.first()).simplify(5e3)
        maskedColl = l8c1t1SR \
            .filterDate(l8compStDt, l8compEndDt) \
            .filterMetadata('CLOUD_COVER', 'less_than', cloudcoverMinThresh) \
            .filterBounds(india.geometry()) \
            .map(maskLS8C1SR)

        return maskedColl

    # Calculate TC coeffs
    def getLS8sr_TCTImage(s2coll, start_doy, end_doy):
        # TCT transform coefficients, for LT8 SR, from Zhai et al. (2022), RSE; Table 1
        # https://doi.org/10.1016/j.rse.2022.112992
        coefficients = ee.Array([
            [ 0.3690,  0.4271,  0.4689, 0.5073,  0.3824,  0.2406],
            [-0.2870, -0.2685, -0.4087, 0.8145,  0.0637, -0.1052],
            [ 0.0382,  0.2137,  0.3536, 0.2270, -0.6108, -0.6351]
        ])

        img = s2coll.filter(ee.Filter.calendarRange(start_doy, end_doy, 'day_of_year')) \
            .median() \
            .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'])

        arrImg1D = img.toArray() # 1D array

        arrImg2D = arrImg1D.toArray(1) # converts 1D array into 6x1 array

        compImg = ee.Image(coefficients) \
            .matrixMultiply(arrImg2D) \
            .arrayProject([0]) \
            .arrayFlatten([['brightness', 'greenness', 'wetness']])

        return compImg

    def createUnionOfMasks(curr, prev):
        currMask = dry.addBands(wet).select([curr]).mask()
        return ee.Image(prev).And(currMask)

    # Load the configuration file and read-in parameters
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configTct = config['FEATURES-TASSELLEDCAP']
    lt8ThreeYrTimeseriesStart = configTct.get('landsat8TCCalc3yrsStartDate')
    lt8ThreeYrTimeseriesEnd   = configTct.get('landsat8TCCalc3yrsEndDate')
    dryStartDoy = configTct.getint('drySeasonStartDoy')
    dryEndDoy   = configTct.getint('drySeasonEndDoy')
    wetStartDoy = configTct.getint('wetSeasonStartDoy')
    wetEndDoy   = configTct.getint('wetSeasonEndDoy')
    dBrtBandName = configTct.get('featureBandNameDrySeasonBrightness')
    dGrnBandName = configTct.get('featureBandNameDrySeasonGreenness')
    dWetBandName = configTct.get('featureBandNameDrySeasonWetness')
    wBrtBandName = configTct.get('featureBandNameWetSeasonBrightness')
    wGrnBandName = configTct.get('featureBandNameWetSeasonGreenness')
    wWetBandName = configTct.get('featureBandNameWetSeasonWetness')
    tctdiffBandName    = configTct.get('featureBandNameTCTDifference')
    tctdiffSegBandName = configTct.get('featureBandNameTCTDifferenceSegmented')
    assetName = configTct.get('existingTasselledcap')
    configAssemble    = config['FEATURES-ASSEMBLE']
    snicRunComp       = configAssemble.getint('snicCompactness')
    snicRunNeighbSize = configAssemble.getint('snicNeighbourhoodSize')
    snicSeedGridFine  = configAssemble.getint('snicSuperPixSeedLocSpacingFINE')
    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    districts = ee.FeatureCollection(config.get('CORE', 'indiaDistricts'))
    trialSpanOfExportDistrictName = config.get('CORE', 'trialSpanToGenerateFeatureRastersOverDistrict')
    trialSpanOfExport = ee.Feature(districts.filter(ee.Filter.eq('DISTRICT', trialSpanOfExportDistrictName)).first())
    fullSpanOfExport  = ee.Feature(ee.FeatureCollection(config.get('CORE', 'fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)

    fine = ee.Algorithms.Image.Segmentation.seedGrid(snicSeedGridFine)
    snicRunParams = {"seeds": fine, "compactness": snicRunComp, "neighborhoodSize": snicRunNeighbSize}

    # Gather band names and prepare to rename the segmentation` outputs
    tctBandnames = [dBrtBandName, dGrnBandName, dWetBandName, wBrtBandName, wGrnBandName, wWetBandName, tctdiffBandName]
    bnWithMeanSuff = [n + '_mean' for n in tctBandnames]
    bnWithSegSuff  = [n + '_seg'  for n in tctBandnames]

    if returnExisting == True:
        tctBands = ee.Image(assetFolder + assetName)
    else:
        # tasseled cap transformed bands for dry and wet seasons for last 3 years
        filteredLS8sr = filterAndMaskCloudsL8C1SR(config)
        LS8sr_3yrs = filteredLS8sr.filterDate(lt8ThreeYrTimeseriesStart, lt8ThreeYrTimeseriesEnd)
        dry = getLS8sr_TCTImage(LS8sr_3yrs, dryStartDoy, dryEndDoy).rename([dBrtBandName, dGrnBandName, dWetBandName]).float()
        wet = getLS8sr_TCTImage(LS8sr_3yrs, wetStartDoy, wetEndDoy).rename([wBrtBandName, wGrnBandName, wWetBandName]).float()

        # Take mask of each of the dry and wet bands and make a union of them.
        # Apply that to them before calculating spectral difference.
        # Avoids -Infinity problems that we detected in final segmented outputs.
        iterFirstIm = ee.Image(1)
        nonsegBands = ee.List([dBrtBandName, dGrnBandName, dWetBandName, wBrtBandName, wGrnBandName, wWetBandName])
        masksUnion = nonsegBands.iterate(createUnionOfMasks, iterFirstIm)
        wet = wet.updateMask(masksUnion)
        dry = dry.updateMask(masksUnion)

        diff = wet.spectralDistance(dry, "sam").rename(tctdiffBandName).float()
        tcts = ee.Image.cat([dry, wet, diff])

        tcts_snicOp = ee.Algorithms.Image.Segmentation.SNIC(** {"image": tcts} | snicRunParams)

        tctBands = tcts.addBands(tcts_snicOp.select(bnWithMeanSuff, bnWithSegSuff)) \
            .set(dict(
                lt8eeid = config.get('FEATURES-TASSELLEDCAP', 'landsat8C1T1SREEID'),
                timeseriesFrom = config.get('FEATURES-TASSELLEDCAP', 'landsat8TCCalc3yrsStartDate'),
                timeseriesTo   = config.get('FEATURES-TASSELLEDCAP', 'landsat8TCCalc3yrsEndDate'),
                snicSegmSeedGridSpacing = snicSeedGridFine,
                snicSegmNeighbSize = snicRunNeighbSize,
                snicCompact = snicRunComp))

        if startFreshExport == True:
            reg = fullSpanOfExport
            # reg = trialSpanOfExport
            # reg = ee.Feature(ee.Geometry.Polygon( \
            #     [[[77.1747772225571, 13.18525174653012], \
            #       [77.1747772225571, 12.39243867438946], \
            #       [81.5253631600571, 12.39243867438946], \
            #       [81.5253631600571, 13.18525174653012]]]), {})
            ee.batch.Export.image.toAsset(** {
              'image': tctBands,
              'description': 'landsat8Tct',
              'assetId': assetFolder + assetName,
              'scale': 30,
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return tctBands

def multiTemporalInterPercentileDifferencesL8(returnExisting = False, startFreshExport = False):
    # normalised difference in inter-percentile difference of NDVI
    def calcNDVI(im):
        return im.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI').float()

    # Load the configuration file and read-in parameters
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configGrnBrnSwings = config['FEATURES-GREENINGTEMPORALSTATS']
    ndviPercs = ast.literal_eval(configGrnBrnSwings.get('greeningPercentilesList'))
    ndviMinBandName = configGrnBrnSwings.get('featureBandNameGreening5thPerc')
    ndviMedBandName = configGrnBrnSwings.get('featureBandNameGreening50thPerc')
    ndviMaxBandName = configGrnBrnSwings.get('featureBandNameGreening95thPerc')
    grnExtBandName    = configGrnBrnSwings.get('featureBandNameGreeningExtent')
    brnExtBandName    = configGrnBrnSwings.get('featureBandNameBrowningExtent')
    grnbrnNdBandName  = configGrnBrnSwings.get('featureBandNameGrningBrningNd')
    ndviMedSegBandName  = configGrnBrnSwings.get('featureBandNameGreening50thPercSegmented')
    grnExtSegBandName   = configGrnBrnSwings.get('featureBandNameGreeningExtentSegmented')
    brnExtSegBandName   = configGrnBrnSwings.get('featureBandNameBrowningExtentSegmented')
    grnbrnNdSegBandName = configGrnBrnSwings.get('featureBandNameGrningBrningNdSegmented')
    configAssemble    = config['FEATURES-ASSEMBLE']
    snicRunComp       = configAssemble.getint('snicCompactness')
    snicRunNeighbSize = configAssemble.getint('snicNeighbourhoodSize')
    snicSeedGridFine  = configAssemble.getint('snicSuperPixSeedLocSpacingFINE')
    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    districts = ee.FeatureCollection(config.get('CORE', 'indiaDistricts'))
    trialSpanOfExportDistrictName = config.get('CORE', 'trialSpanToGenerateFeatureRastersOverDistrict')
    trialSpanOfExport = ee.Feature(districts.filter(ee.Filter.eq('DISTRICT', trialSpanOfExportDistrictName)).first())
    fullSpanOfExport  = ee.Feature(ee.FeatureCollection(config.get('CORE', 'fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)

    fine = ee.Algorithms.Image.Segmentation.seedGrid(snicSeedGridFine)
    snicRunParams = {"seeds": fine, "compactness": snicRunComp, "neighborhoodSize": snicRunNeighbSize}

    if returnExisting == True:
        assetName = configGrnBrnSwings.get('existingGreening')
        ndvi_stats = ee.Image(assetFolder + assetName)
    else:
        # Calculate percentile stats over time of NDVI
        multitemporalNDVI = filterAndMaskCloudsL8C2SR(config).map(calcNDVI)
        ndvi_stats = multitemporalNDVI.select('NDVI') \
            .reduce(ee.Reducer.percentile([5, 50, 95])) \
            .rename([ndviMinBandName, ndviMedBandName, ndviMaxBandName]) \
            .float()
        # Calculate greening and browning extents: mx-md, md-mn
        ndvi_stats = ndvi_stats.addBands([
            ndvi_stats.select(ndviMaxBandName).subtract(ndvi_stats.select(ndviMedBandName)).rename(grnExtBandName),
            ndvi_stats.select(ndviMedBandName).subtract(ndvi_stats.select(ndviMinBandName)).rename(brnExtBandName)
            ])
        # Calculate normalized diff of greening and browning extents
        # From the original notebook, dropping .select(['gr_ext', 'br_ext', 'nd_grbr']) at the end
        # because the next step of final composite generation & exporting expects all the bands.
        ndvi_stats = ndvi_stats.addBands([
            ndvi_stats.normalizedDifference([grnExtBandName, brnExtBandName]).rename(grnbrnNdBandName)
            ])

        ndvi = ndvi_stats.select(ndviMedBandName).float()
        brng = ndvi_stats.select(brnExtBandName).float()
        grng = ndvi_stats.select(grnExtBandName).float()
        nd_grbr = ndvi_stats.select(grnbrnNdBandName).float()

        ndvi_seg = ee.Algorithms.Image.Segmentation.SNIC(** {"image": ndvi} | snicRunParams).select(ndviMedBandName + "_mean").rename(ndviMedSegBandName)
        brng_seg = ee.Algorithms.Image.Segmentation.SNIC(** {"image": brng} | snicRunParams).select(brnExtBandName + "_mean").rename(brnExtSegBandName)
        grng_seg = ee.Algorithms.Image.Segmentation.SNIC(** {"image": grng} | snicRunParams).select(grnExtBandName + "_mean").rename(grnExtSegBandName)
        nd_grbr_seg = ee.Algorithms.Image.Segmentation.SNIC(** {"image": nd_grbr} | snicRunParams).select(grnbrnNdBandName + "_mean").rename(grnbrnNdSegBandName)

        ndvi_stats = ndvi_stats.addBands(ee.Image.cat(ndvi_seg, brng_seg, grng_seg, nd_grbr_seg)) \
            .set(dict(
                    lt8eeid = config.get('FEATURES-SEASONALITY', 'landsat8C2T1SREEID'),
                    timeseriesFrom = config.get('FEATURES-SEASONALITY', 'landsat8CompositingStartDate'),
                    timeseriesTo   = config.get('FEATURES-SEASONALITY', 'landsat8CompositingEndDate'),
                    snicSegmSeedGridSpacing = snicSeedGridFine,
                    snicSegmNeighbSize = snicRunNeighbSize,
                    snicCompact = snicRunComp))

        if startFreshExport == True:
            reg = fullSpanOfExport
            # reg = trialSpanOfExport
            ee.batch.Export.image.toAsset(** {
              'image': ndvi_stats,
              'description': 'landsat8GreenBrownSwings',
              'assetId': assetFolder + 'lt8GreenBrownSwings',
              'scale': 30,
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return ndvi_stats

def palsarAggregation(returnExisting = False, startFreshExport = False):
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configPalsar = config['FEATURES-PALSAR']
    palsar = ee.ImageCollection(configPalsar.get('palsarEEID'))
    aggrStartDate = configPalsar.get('startDate')
    aggrEndDate   = configPalsar.get('endDate')
    medBandName = configPalsar.get('featureBandNamePalsarMedian')
    medSegBandName = configPalsar.get('featureBandNamePalsarMedianSegmented')
    configAssemble    = config['FEATURES-ASSEMBLE']
    snicRunComp       = configAssemble.getint('snicCompactness')
    snicRunNeighbSize = configAssemble.getint('snicNeighbourhoodSize')
    snicSeedGridFine  = configAssemble.getint('snicSuperPixSeedLocSpacingFINE')
    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    districts = ee.FeatureCollection(config.get('CORE', 'indiaDistricts'))
    trialSpanOfExportDistrictName = config.get('CORE', 'trialSpanToGenerateFeatureRastersOverDistrict')
    trialSpanOfExport = ee.Feature(districts.filter(ee.Filter.eq('DISTRICT', trialSpanOfExportDistrictName)).first())
    fullSpanOfExport  = ee.Feature(ee.FeatureCollection(config.get('CORE', 'fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)

    fine = ee.Algorithms.Image.Segmentation.seedGrid(snicSeedGridFine)
    snicRunParams = {"seeds": fine, "compactness": snicRunComp, "neighborhoodSize": snicRunNeighbSize}

    if returnExisting == True:
        assetName = configPalsar.get('existingPalsarVeg')
        palsarMd = ee.Image(assetFolder + assetName)
    else:
        palsarMd = palsar.filterDate(aggrStartDate, aggrEndDate) \
            .select('HV') \
            .median() \
            .rename(medBandName) \
            .float()
        palsar_seg = ee.Algorithms.Image.Segmentation.SNIC(** {"image": palsarMd} | snicRunParams).select(medBandName + "_mean").rename(medSegBandName)
        palsarMd = palsarMd.addBands(palsar_seg) \
            .set(dict(
                timeseriesFrom = aggrStartDate,
                timeseriesTo = aggrEndDate,
                snicSegmSeedGridSpacing = snicSeedGridFine,
                snicSegmNeighbSize = snicRunNeighbSize,
                snicCompact = snicRunComp))

        if startFreshExport == True:
            reg = fullSpanOfExport
            # reg = trialSpanOfExport
            ee.batch.Export.image.toAsset(** {
              'image': palsarMd,
              'description': 'palsarVegDens',
              'assetId': assetFolder + 'palsarVegDens',
              'scale': 30,
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return palsarMd

def evapotranspirationAggregation(returnExisting = False, startFreshExport = False):
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configEt = config['FEATURES-EVAPOTRANSPIRATION']
    et = ee.ImageCollection(configEt.get('evapotranspirationEEID')) \
        .select(ast.literal_eval(configEt.get('evapotranspirationBandnames')))
    etYear = config.getint('FEATURES-EVAPOTRANSPIRATION', 'evapotranspirationYear')
    etSoilBandName = configEt.get('featureBandNameEtSoil')
    etVegBandName  = configEt.get('featureBandNameEtVeg')
    etSoilSegBandName = configEt.get('featureBandNameEtSoilSegmented')
    etVegSegBandName  = configEt.get('featureBandNameEtVegSegmented')
    configAssemble    = config['FEATURES-ASSEMBLE']
    snicRunComp       = configAssemble.getint('snicCompactness')
    snicRunNeighbSize = configAssemble.getint('snicNeighbourhoodSize')
    snicSeedGridFine  = configAssemble.getint('snicSuperPixSeedLocSpacingFINE')
    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    districts = ee.FeatureCollection(config.get('CORE', 'indiaDistricts'))
    trialSpanOfExportDistrictName = config.get('CORE', 'trialSpanToGenerateFeatureRastersOverDistrict')
    trialSpanOfExport = ee.Feature(districts.filter(ee.Filter.eq('DISTRICT', trialSpanOfExportDistrictName)).first())
    fullSpanOfExport  = ee.Feature(ee.FeatureCollection(config.get('CORE', 'fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)

    fine = ee.Algorithms.Image.Segmentation.seedGrid(snicSeedGridFine)
    snicRunParams = {"seeds": fine, "compactness": snicRunComp, "neighborhoodSize": snicRunNeighbSize}

    if returnExisting == True:
        assetName = configEt.get('existingEt')
        etAnnual = ee.Image(assetFolder + assetName)
    else:
        etYearStartDate = ee.Date.fromYMD(etYear, 1, 1)
        etAnnual = et.filterDate(etYearStartDate, etYearStartDate.advance(1, 'year')) \
            .sum() \
            .float() \
            .rename([etVegBandName, etSoilBandName])
        #     .resample('bilinear') \ before rename
        et_seg = ee.Algorithms.Image.Segmentation.SNIC(** {"image": etAnnual} | snicRunParams).select([etSoilBandName + "_mean", etVegBandName + "_mean"]).rename([etSoilSegBandName, etVegSegBandName])
        etAnnual = etAnnual.addBands(et_seg).set(dict(
            etYear = etYear,
            snicSegmSeedGridSpacing = snicSeedGridFine,
            snicSegmNeighbSize = snicRunNeighbSize,
            snicCompact = snicRunComp))

        if startFreshExport == True:
            reg = fullSpanOfExport
            # reg = trialSpanOfExport
            ee.batch.Export.image.toAsset(** {
              'image': etAnnual,
              'description': 'et',
              'assetId': assetFolder + 'etAnnual',
              'scale': 30,
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return etAnnual

def elevationAggregation(returnExisting = False, startFreshExport = False):
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configElev = config['FEATURES-ELEVATION']
    elev            = ee.Image(configElev.get('demEEID'))
    elevBand        = configElev.get('demBandName')
    elevFeatureName = configElev.get('featureBandName')
    elevSegFeatureName = configElev.get('featureBandNameElevSegmented')
    configAssemble    = config['FEATURES-ASSEMBLE']
    snicRunComp       = configAssemble.getint('snicCompactness')
    snicRunNeighbSize = configAssemble.getint('snicNeighbourhoodSize')
    snicSeedGridFine  = configAssemble.getint('snicSuperPixSeedLocSpacingFINE')
    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    districts = ee.FeatureCollection(config.get('CORE', 'indiaDistricts'))
    trialSpanOfExportDistrictName = config.get('CORE', 'trialSpanToGenerateFeatureRastersOverDistrict')
    trialSpanOfExport = ee.Feature(districts.filter(ee.Filter.eq('DISTRICT', trialSpanOfExportDistrictName)).first())
    fullSpanOfExport  = ee.Feature(ee.FeatureCollection(config.get('CORE', 'fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)

    fine = ee.Algorithms.Image.Segmentation.seedGrid(snicSeedGridFine)
    snicRunParams = {'seeds': fine, 'compactness': snicRunComp, 'neighborhoodSize': snicRunNeighbSize}

    if returnExisting == True:
        assetName = configElev.get('existingElevation')
        elev_seg = ee.Image(assetFolder + assetName)
    else:
        elev = elev.select(elevBand) \
            .rename(elevFeatureName) \
            .float()

        elev_seg = ee.Algorithms.Image.Segmentation.SNIC(** {'image': elev} | snicRunParams).select(elevFeatureName + "_mean").rename(elevSegFeatureName) \
            .set(dict(
                snicSegmSeedGridSpacing = snicSeedGridFine,
                snicSegmNeighbSize = snicRunNeighbSize,
                snicCompact = snicRunComp))

        if startFreshExport == True:
            reg = fullSpanOfExport
            # reg = trialSpanOfExport
            ee.batch.Export.image.toAsset(** {
              'image': elev_seg,
              'description': 'elev',
              'assetId': assetFolder + 'elevation',
              'scale': 30,
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return elev_seg

def precipitationAggregation(returnExisting = False, startFreshExport = False):
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configPrecip = config['FEATURES-PRECIPITATION']
    ppt            = ee.Image(configPrecip.get('worldclimBioEEID'))
    pptBand        = configPrecip.get('worldclimBioAnnPrecipBandname')
    pptFeatureName = configPrecip.get('featureBandName')
    pptSegFeatureName = configPrecip.get('featureBandNameAnnRainSegmented')
    configAssemble    = config['FEATURES-ASSEMBLE']
    snicRunComp       = configAssemble.getint('snicCompactness')
    snicRunNeighbSize = configAssemble.getint('snicNeighbourhoodSize')
    snicSeedGridFine  = configAssemble.getint('snicSuperPixSeedLocSpacingFINE')
    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    districts = ee.FeatureCollection(config.get('CORE', 'indiaDistricts'))
    trialSpanOfExportDistrictName = config.get('CORE', 'trialSpanToGenerateFeatureRastersOverDistrict')
    trialSpanOfExport = ee.Feature(districts.filter(ee.Filter.eq('DISTRICT', trialSpanOfExportDistrictName)).first())
    fullSpanOfExport  = ee.Feature(ee.FeatureCollection(config.get('CORE', 'fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)

    fine = ee.Algorithms.Image.Segmentation.seedGrid(snicSeedGridFine)
    snicRunParams = {'seeds': fine, 'compactness': snicRunComp, 'neighborhoodSize': snicRunNeighbSize}

    if returnExisting == True:
        assetName = configPrecip.get('existingPrecipAvgAnnual')
        ppt_seg = ee.Image(assetFolder + assetName)
    else:
        ppt = ppt.select(pptBand) \
            .rename(pptFeatureName) \
            .float()

        ppt_seg = ee.Algorithms.Image.Segmentation.SNIC(** {'image': ppt} | snicRunParams).select(pptFeatureName + "_mean").rename(pptSegFeatureName) \
            .set(dict(
                snicSegmSeedGridSpacing = snicSeedGridFine,
                snicSegmNeighbSize = snicRunNeighbSize,
                snicCompact = snicRunComp))

        if startFreshExport == True:
            reg = fullSpanOfExport
            # reg = trialSpanOfExport
            ee.batch.Export.image.toAsset(** {
              'image': ppt_seg,
              'description': 'precipAvgAnnual',
              'assetId': assetFolder + 'precipAvgAnnual',
              'scale': 30,
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return ppt_seg

def topographyMtpiAggregation(returnExisting = False, startFreshExport = False):
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configTopoMtpi = config['FEATURES-TOPOGRAPHY-MTPI']
    topo            = ee.Image(configTopoMtpi.get('topographicPositionIndexEEID'))
    topoBand        = configTopoMtpi.get('mtpiBandname')
    topoFeatureName = configTopoMtpi.get('featureBandName')
    topoSegFeatureName = configTopoMtpi.get('featureBandNameTopoMtpiSegmented')
    configAssemble    = config['FEATURES-ASSEMBLE']
    snicRunComp       = configAssemble.getint('snicCompactness')
    snicRunNeighbSize = configAssemble.getint('snicNeighbourhoodSize')
    snicSeedGridFine  = configAssemble.getint('snicSuperPixSeedLocSpacingFINE')
    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    districts = ee.FeatureCollection(config.get('CORE', 'indiaDistricts'))
    trialSpanOfExportDistrictName = config.get('CORE', 'trialSpanToGenerateFeatureRastersOverDistrict')
    trialSpanOfExport = ee.Feature(districts.filter(ee.Filter.eq('DISTRICT', trialSpanOfExportDistrictName)).first())
    fullSpanOfExport  = ee.Feature(ee.FeatureCollection(config.get('CORE', 'fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)

    fine = ee.Algorithms.Image.Segmentation.seedGrid(snicSeedGridFine)
    snicRunParams = {'seeds': fine, 'compactness': snicRunComp, 'neighborhoodSize': snicRunNeighbSize}

    if returnExisting == True:
        assetName = configTopoMtpi.get('existingTopoMtpi')
        topo_seg = ee.Image(assetFolder + assetName)
    else:
        topo = topo.select(topoBand) \
            .rename(topoFeatureName) \
            .float()

        topo_seg = ee.Algorithms.Image.Segmentation.SNIC(** {'image': topo} | snicRunParams).select(topoFeatureName + "_mean").rename(topoSegFeatureName) \
            .set(dict(
                snicSegmSeedGridSpacing = snicSeedGridFine,
                snicSegmNeighbSize = snicRunNeighbSize,
                snicCompact = snicRunComp))

        if startFreshExport == True:
            reg = fullSpanOfExport
            # reg = trialSpanOfExport
            ee.batch.Export.image.toAsset(** {
              'image': topo_seg,
              'description': 'topoMtpi',
              'assetId': assetFolder + 'topoMtpi',
              'scale': 30,
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return topo_seg

def topographyHandAggregation(returnExisting = False, startFreshExport = False):
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configTopoHand = config['FEATURES-TOPOGRAPHY-HAND']
    hand30_100          = ee.ImageCollection(configTopoHand.get('heightAboveNearestDrainageEEID'))
    topoHandBand        = configTopoHand.get('handBandname')
    topoHandFeatureName = configTopoHand.get('featureBandName')
    topoHandSegFeatureName = configTopoHand.get('featureBandNameTopoHandSegmented')
    configAssemble    = config['FEATURES-ASSEMBLE']
    snicRunComp       = configAssemble.getint('snicCompactness')
    snicRunNeighbSize = configAssemble.getint('snicNeighbourhoodSize')
    snicSeedGridFine  = configAssemble.getint('snicSuperPixSeedLocSpacingFINE')
    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    districts = ee.FeatureCollection(config.get('CORE', 'indiaDistricts'))
    trialSpanOfExportDistrictName = config.get('CORE', 'trialSpanToGenerateFeatureRastersOverDistrict')
    trialSpanOfExport = ee.Feature(districts.filter(ee.Filter.eq('DISTRICT', trialSpanOfExportDistrictName)).first())
    fullSpanOfExport  = ee.Feature(ee.FeatureCollection(config.get('CORE', 'fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)

    fine = ee.Algorithms.Image.Segmentation.seedGrid(snicSeedGridFine)
    snicRunParams = {'seeds': fine, 'compactness': snicRunComp, 'neighborhoodSize': snicRunNeighbSize}

    if returnExisting == True:
        assetName = configTopoHand.get('existingTopoHand')
        topo_hand_seg = ee.Image(assetFolder + assetName)
    else:
        topo_hand = hand30_100.mosaic().select(topoHandBand) \
            .rename(topoHandFeatureName) \
            .float()

        topo_hand_seg = ee.Algorithms.Image.Segmentation.SNIC(** {'image': topo_hand} | snicRunParams).select(topoHandFeatureName + "_mean").rename(topoHandSegFeatureName) \
            .set(dict(
                snicSegmSeedGridSpacing = snicSeedGridFine,
                snicSegmNeighbSize = snicRunNeighbSize,
                snicCompact = snicRunComp))

        if startFreshExport == True:
            reg = fullSpanOfExport
            # reg = trialSpanOfExport
            ee.batch.Export.image.toAsset(** {
              'image': topo_hand_seg,
              'description': 'topoHand',
              'assetId': assetFolder + 'topoHand',
              'scale': 30,
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return topo_hand_seg

def geomorphologyTerrainRuggednessAggregation(returnExisting = False, startFreshExport = False):
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configGeomorphRugg = config['FEATURES-TOPOGRAPHY-GEOMORPHOLOGY-RUGGEDNESS']
    ruggednessIndex    = ee.ImageCollection(configGeomorphRugg.get('terrainRuggednessIndexEEID'))
    ruggednessBand     = configGeomorphRugg.get('ruggedBandname')
    ruggednessFeatureName = configGeomorphRugg.get('featureBandName')
    ruggednessSegFeatureName = configGeomorphRugg.get('featureBandNameGeomorphRuggedSegmented')
    assetName = configGeomorphRugg.get('existingGeomorphRugged')
    configAssemble    = config['FEATURES-ASSEMBLE']
    snicRunComp       = configAssemble.getint('snicCompactness')
    snicRunNeighbSize = configAssemble.getint('snicNeighbourhoodSize')
    snicSeedGridFine  = configAssemble.getint('snicSuperPixSeedLocSpacingFINE')
    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    districts = ee.FeatureCollection(config.get('CORE', 'indiaDistricts'))
    trialSpanOfExportDistrictName = config.get('CORE', 'trialSpanToGenerateFeatureRastersOverDistrict')
    trialSpanOfExport = ee.Feature(districts.filter(ee.Filter.eq('DISTRICT', trialSpanOfExportDistrictName)).first())
    fullSpanOfExport  = ee.Feature(ee.FeatureCollection(config.get('CORE', 'fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)

    fine = ee.Algorithms.Image.Segmentation.seedGrid(snicSeedGridFine)
    snicRunParams = {'seeds': fine, 'compactness': snicRunComp, 'neighborhoodSize': snicRunNeighbSize}

    if returnExisting == True:
        geomorphRugg_ops = ee.Image(assetFolder + assetName)
    else:
        geomorphRugg = ruggednessIndex \
            .filterBounds(fullSpanOfExport.geometry()) \
            .select(ruggednessBand) \
            .mosaic() \
            .rename(ruggednessFeatureName) \
            .float()

        geomorphRugg_seg = ee.Algorithms.Image.Segmentation.SNIC(** {'image': geomorphRugg} | snicRunParams).select(ruggednessFeatureName + "_mean").rename(ruggednessSegFeatureName)

        geomorphRugg_ops = ee.Image.cat(geomorphRugg, geomorphRugg_seg) \
            .set(dict(
                snicSegmSeedGridSpacing = snicSeedGridFine,
                snicSegmNeighbSize = snicRunNeighbSize,
                snicCompact = snicRunComp))

        if startFreshExport == True:
            reg = fullSpanOfExport
            # reg = trialSpanOfExport
            ee.batch.Export.image.toAsset(** {
              'image': geomorphRugg_ops,
              'description': 'geomorphRuggedness',
              'assetId': assetFolder + assetName,
              'scale': 30,
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return geomorphRugg_ops

def assembleAllExistingFeatureRasters():
    pheno = seasonalityParamsL8(returnExisting = True)
    tctd = tasselledCapCoeffsL8(returnExisting = True)
    ndvp = multiTemporalInterPercentileDifferencesL8(returnExisting = True)
    palsar = palsarAggregation(returnExisting = True)
    et = evapotranspirationAggregation(returnExisting = True)
    elev = elevationAggregation(returnExisting = True)
    ppt = precipitationAggregation(returnExisting = True)
    topo = topographyMtpiAggregation(returnExisting = True)
    topo_hand = topographyHandAggregation(returnExisting = True)
    geomorph = geomorphologyTerrainRuggednessAggregation(returnExisting = True)
    zonesStatesNum = semiarid.classificationZonesFromStatesNumeric(returnExisting = True)
    zonesStatesOhe = semiarid.classificationZonesFromStatesOneHotEncoded(returnExisting = True)
    zonesBiomesNum = semiarid.classificationZonesFromBiomesNumeric(returnExisting = True)
    zonesBiomesOhe = semiarid.classificationZonesFromBiomesOneHotEncoded(returnExisting = True)
    zonesGeoAgeNum = semiarid.classificationZonesFromGeologicalAgeNumeric(returnExisting = True)
    zonesGeoAgeOhe = semiarid.classificationZonesFromGeologicalAgeOneHotEncoded(returnExisting = True)
    aoiMask = semiarid.maskWithClassLabels(returnExisting = True)
    lon = ee.Image.pixelLonLat().select('longitude').float()
    lat = ee.Image.pixelLonLat().select('latitude').float()

    assembled = ee.Image.cat(pheno, tctd, ndvp, palsar, et, elev, ppt, topo, topo_hand, geomorph, lon, lat, zonesStatesNum, zonesStatesOhe, zonesBiomesNum, zonesBiomesOhe, zonesGeoAgeNum, zonesGeoAgeOhe, aoiMask)

    return assembled

def assembleFeatureBandsAndExport(returnExisting = False, startFreshExport = False):
    def sampleFeatures(pt):
        samples = allFeatures.reduceRegion(** {
            "reducer": ee.Reducer.first(),
            "geometry": pt.geometry(),
            "scale": 30,
            "tileScale": 8
        })
        labelNumeric     = ee.Dictionary({    "labelNum": allONEMulticlassLabelnamesAndTheirNumLabels.getNumber(pt.getString("label"))})
        labelPalsHarmNumeric = ee.Dictionary({"label_palsarHarmonisedNum": allONEMulticlassPalsarHarmonizedLabelnamesAndTheirNumLabels.getNumber(pt.getString("lulcLabel"))})
        onelabelNumeric  = ee.Dictionary({      "oneNum": allONESingleclassLabelnamesAndTheirNumLabels.getNumber(pt.getString( "one"))})
        lulclabelNumeric = ee.Dictionary({"lulc_codeNum": allLULCcodeaLabelnamesAndTheirNumLabels.getNumber(pt.getString("lulc_code"))})
        wlalabelNumeric  = ee.Dictionary({"wla_codeNum": allWLAcodesLabelnamesAndTheirNumLabels.getNumber(pt.getString(   "wla_code"))})
        return pt.set(samples.combine(labelNumeric).combine(labelPalsHarmNumeric).combine(lulclabelNumeric).combine(wlalabelNumeric).combine(onelabelNumeric))

    # Load the configuration file and read-in parameters
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    origTrainingPoints = ee.FeatureCollection(config.get('CORE', 'lulcLabeledPoints'))
    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    assetName   = config.get('FEATURES-ASSEMBLE', 'existingLabeledPointsWithFeatures')
    trptsAssetsSubfolder   = config.get('FEATURES-ASSEMBLE', 'subfolderForPointsWithFeatures')
    oneMulticlassLabels  = ast.literal_eval(config.get('FEATURES-ASSEMBLE', 'lulcLabelsEnhancedForONEMulticlass'))
    oneMulticlassLabelsPalsarHarmonized  = ast.literal_eval(config.get('FEATURES-ASSEMBLE', 'lulcLabelsPalsarHarmonizedForONEMulticlass'))
    oneSingleclassLabels = ast.literal_eval(config.get('FEATURES-ASSEMBLE', 'lulcLabelsForONE'))
    lulccodesLabels      = ast.literal_eval(config.get('FEATURES-ASSEMBLE', 'lulcLabelsForlulcCodes'))
    wlacodesLabels       = ast.literal_eval(config.get('FEATURES-ASSEMBLE', 'lulclabelsForWastelandAtlasCodes'))
    districts = ee.FeatureCollection(config.get('CORE', 'indiaDistricts'))
    trialSpanOfExportDistrictName = config.get('CORE', 'trialSpanToGenerateFeatureRastersOverDistrict')
    trialSpanOfExport = ee.Feature(districts.filter(ee.Filter.eq('DISTRICT', trialSpanOfExportDistrictName)).first())
    fullSpanOfExport  = ee.Feature(ee.FeatureCollection(config.get('CORE', 'fullSpanToGenerateFeatureRastersOver')).first()).simplify(500)

    if returnExisting == True:
        # Run shell command to get the list of IDs for all the state-wise point tables
        # Ref: https://stackoverflow.com/a/4256153
        cmdToListStatewisePointsTableIDs = f"earthengine ls {assetFolder}{trptsAssetsSubfolder}"
        process = subprocess.Popen(cmdToListStatewisePointsTableIDs.split(), stdout=subprocess.PIPE)
        statewisePointsTableIDsBytes, error = process.communicate()

        # The list is in the form of bytes with names separated by "\n"
        # Split by "\n" to get a list of bytes, drop last element since empty ""
        # Ref: https://stackoverflow.com/a/15095537; https://stackoverflow.com/a/18170012
        statewisePtsIDsList = statewisePointsTableIDsBytes.split(b'\n')
        del statewisePtsIDsList[-1]

        # Run through the list and gather the tables into one merged featurecollection
        fc = ee.FeatureCollection([])
        for stateTableID in statewisePtsIDsList:
            # Ref: https://www.geeksforgeeks.org/how-to-convert-bytes-to-string-in-python/
            fc = fc.merge(ee.FeatureCollection(stateTableID.decode()))

        sampledPoints = fc
    else:
        # Gather up ALL feature rasters, sample them
        allFeatures = assembleAllExistingFeatureRasters()
        reg = fullSpanOfExport
        # reg = trialSpanOfExport

        # To encode the label columns with string values
        # (ONE multiclass, original lulc codes, WLA codes and ONE y/n)
        # into numeric values, prepare dictionaries with labelstring-numeral pairings
        # for each case
        labelNameListONEMulticlass = ee.List(oneMulticlassLabels)
        labelValListONEMulticlass = [i for i in range(1, len(oneMulticlassLabels)+1)]
        allONEMulticlassLabelnamesAndTheirNumLabels = ee.Dictionary.fromLists(labelNameListONEMulticlass, labelValListONEMulticlass)

        labelNameListONEMulticlassPalsarHarmonized = ee.List(oneMulticlassLabelsPalsarHarmonized)
        labelValListONEMulticlassPalsarHarmonized = [i for i in range(1, len(oneMulticlassLabelsPalsarHarmonized)+1)]
        allONEMulticlassPalsarHarmonizedLabelnamesAndTheirNumLabels = ee.Dictionary.fromLists(labelNameListONEMulticlassPalsarHarmonized, labelValListONEMulticlassPalsarHarmonized)

        labelNameListLULCcodes = ee.List(lulccodesLabels)
        labelValListLULCcodes = [i for i in range(1, len(lulccodesLabels)+1)]
        allLULCcodeaLabelnamesAndTheirNumLabels = ee.Dictionary.fromLists(labelNameListLULCcodes, labelValListLULCcodes)

        labelNameListWLAcodes = ee.List(wlacodesLabels)
        labelValListWLAcodes = [i for i in range(1, len(wlacodesLabels)+1)]
        allWLAcodesLabelnamesAndTheirNumLabels = ee.Dictionary.fromLists(labelNameListWLAcodes, labelValListWLAcodes)

        labelNameListONESingleclass = ee.List(oneSingleclassLabels)
        labelValListONESingleclass = [i for i in range(1, len(oneSingleclassLabels)+1)]
        allONESingleclassLabelnamesAndTheirNumLabels = ee.Dictionary.fromLists(labelNameListONESingleclass, labelValListONESingleclass)

        # Get the list of states from the points table
        # and run the sampling+export on each state separately
        trptsStates = origTrainingPoints.aggregate_array("state").distinct().getInfo()
        for state in trptsStates:
            # Filter points to state, sample and export as separate table.
            # Add a random column, to help with debugging, etc., with a smaller sampling of points.
            pointsToSample = origTrainingPoints.filter(ee.Filter.eq("state", state)) \
                .randomColumn("statewiserandForFewerPts")
            print(state + " num points ", pointsToSample.size().getInfo())
            sampledPoints = pointsToSample.map(sampleFeatures)

            if startFreshExport == True:
                ee.batch.Export.table.toAsset(** {
                  'collection': sampledPoints,
                  'description': assetName + "_" + state,
                  'assetId': assetFolder + trptsAssetsSubfolder + "/" + assetName + "_" + state
                }).start()

    return sampledPoints
