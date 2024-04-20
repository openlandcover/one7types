import ee
import configparser as cp
import ast
import io

def maskWithClassLabels(returnExisting = False, startFreshExport = False):
    labelValid   = 1
    labelInvalid = 0
    labelForest  = 2

    # Load the configuration file and read-in parameters
    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    configCore = config["CORE"]
    assetFolder = configCore.get('assetFolderWithFeatures')
    namesOfStatesOfInterest = ast.literal_eval(configCore.get("allStatesOfInterest"))
    mhgoaExceptions = ast.literal_eval(configCore.get("statesException1"))
    belgaumExceptions = ast.literal_eval(configCore.get("districtsException1"))
    configAoi = config["AOI"]
    assetName = configAoi.get("existingAOI")
    bandName  = configAoi.get("bandNameAOI")
    statesFc = ee.FeatureCollection(configAoi.get('indiaStatesID'))
    districtsFc = ee.FeatureCollection(configAoi.get('indiaDistrictsID'))
    elev = ee.Image(configAoi.get('demEEID'))
    annualPrecip = ee.Image(configAoi.get('worldclimBioEEID')) \
        .select(configAoi.get('worldclimBioAnnPrecipBandname'))
    urbanAreasFromNighttimeLights = ee.ImageCollection(configAoi.get('nighttimeLightsEEID')) \
        .filterDate('2019-11-01', '2020-08-01') \
        .select(configAoi.get('nighttimeLightsIntensityBandname')) \
        .max()
    humanSettlements = ee.Image(configAoi.get('humanSettlement2016EEID')) \
        .select(configAoi.get('humanSettlement2016Bandname')) \
        .neq(configAoi.getint('humanSettlement2016LabelLandnobuiltup'))
    water = ee.Image(configAoi.get('surfaceWaterSummaryEEID')) \
        .select(configAoi.get('surfaceWaterMaxextentBandname')) \
        .eq(configAoi.getint('surfaceWaterMaxextentLabelWater'))
    # From the full collection of PALSAR FnF, where ever there was forest
    forest = ee.ImageCollection(configAoi.get('palsarForestEEID')) \
        .map(lambda image: image.select(configAoi.get('palsarForestBandname')).eq(config.getint('AOI', 'palsarForestLabelForest')) ) \
        .max()
    gjClipCoords = ast.literal_eval(configAoi.get("gujaratAreaPoly"))
    gujarat_clip = ee.Geometry.MultiPolygon(gjClipCoords)
    excludePolyCoords = ast.literal_eval(configAoi.get("excludePoly"))
    exclude = ee.Geometry.MultiPolygon(excludePolyCoords)
    includePolyCoords = ast.literal_eval(configAoi.get("includePoly"))
    include = ee.Geometry.MultiPolygon(includePolyCoords)

    if returnExisting == True:
        inputMask = ee.Image(assetFolder + assetName)
    else:
        # Generate the aoi mask
        statesOfInterest = statesFc \
            .filter(ee.Filter.inList('state', namesOfStatesOfInterest));
        mhgoa = statesFc.filter(ee.Filter.inList('state', mhgoaExceptions))
        belgaum = districtsFc.filter(ee.Filter.inList('dtname', belgaumExceptions))

        inputMask = ee.Image(0).clipToCollection(statesOfInterest) \
            .where(elev.gt(500).clipToCollection(mhgoa), labelValid) \
            .where(elev.clip(gujarat_clip).gt(100), labelValid) \
            .where(elev.gt(600).clipToCollection(belgaum), labelValid) \
            .where(annualPrecip.lte(1200).clipToCollection(statesOfInterest), labelValid) \
            .where(ee.Image(1).clip(include), labelValid) \
            .where(ee.Image(1).clip(exclude), labelInvalid) \
            .where(urbanAreasFromNighttimeLights.gte(4), labelInvalid) \
            .where(humanSettlements, labelInvalid) \
            .where(water, labelInvalid) \
            .selfMask() \
            .multiply(0) \
            .where(forest, labelForest) \
            .rename(bandName)

        if startFreshExport == True:
            reg = statesOfInterest.geometry().bounds()
            ee.batch.Export.image.toAsset(** {
              "image": inputMask,
              "description": assetName,
              "assetId": assetFolder + assetName,
              "region": reg,
              "scale": 30,
              "maxPixels": 1e12,
              "pyramidingPolicy": {".default": "mode"}
            }).start()

    return inputMask

def classificationZonesFromRegionsFeatureCollection(fc = None, zoneBandNamePrefix = None, oneHotEncoded = False):
    def generateOneHotEncodedImagePerZone(f):
        zn = ee.Feature(f).getNumber("zoneNum")
        # Have to do -1 on zn each time, because zoneNumberSeq starts from 1
        indexOfznIntoLists = zn.subtract(1)
        return ee.Image(0) \
            .where(zonesNumeric.eq(zn), 1) \
            .rename(ee.String(zoneBandNamePrefix + "Ohe_").cat(f.getString("zoneLabel")))

    zonesNumeric = fc.reduceToImage(["zoneNum"], ee.Reducer.first()) \
        .toUint8() \
        .rename(zoneBandNamePrefix + "Num")

    classifZones = zonesNumeric

    if oneHotEncoded == True:
        # Create a 1/0 image per zone
        oheIm = ee.ImageCollection(fc.map(generateOneHotEncodedImagePerZone)).toBands();
        # Remove the system id prefix toBands() adds to each band's name
        oheImBands = oheIm.bandNames()
        oheImBandsSysIdPrefixRemoved = oheImBands.map(lambda n: ee.String(n).slice(ee.String(n).index("_").add(1)))
        zonesOhe = oheIm.select(oheImBands, oheImBandsSysIdPrefixRemoved)

        classifZones = zonesOhe

    return classifZones

def classificationZonesFromStatesNumeric(returnExisting = False, startFreshExport = False):
    def assignNumberToStateGroupZone(i):
        # Have to do -1 on i each time, because zoneNumberSeq starts from 1
        indexOfiIntoLists = ee.Number(i).subtract(1)
        zoneOfi = ee.Feature(allStates.filter(ee.Filter.inList("state", ee.List(stateGroupingsForZones.get(indexOfiIntoLists)))).union().first())
        return zoneOfi.set("zoneNum", zoneNumberSeq.getNumber(indexOfiIntoLists), "zoneLabel", zoneLabels.getString(indexOfiIntoLists))

    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    allStates = ee.FeatureCollection(config.get("CORE", "indiaStates"))
    configZonesStates = config["AOI-CLASSIFICATION-ZONES-STATES"]
    assetName   = configZonesStates.get("existingZonesNumeric")
    stateGroupingsForZones = ee.List(ast.literal_eval(configZonesStates.get("groupsOfStates")))
    zoneLabels             = ee.List(ast.literal_eval(configZonesStates.get("groupsOfStatesLabels")))
    zoneBandnamePrefix = configZonesStates.get("featureBandNamePrefix")

    if returnExisting == True:
        zones = ee.Image(assetFolder + assetName)
    else:
        # Group states into zones, assign a numeric and a string label to
        zoneNumberSeq = ee.List.sequence(1, zoneLabels.length());
        statesgroupsWithZoneNumberAssigned = ee.FeatureCollection(zoneNumberSeq.map(assignNumberToStateGroupZone))
        statesZonesNum = classificationZonesFromRegionsFeatureCollection(statesgroupsWithZoneNumberAssigned, zoneBandnamePrefix, oneHotEncoded = False)
        zones = statesZonesNum

        if startFreshExport == True:
            reg = ee.Feature(ee.FeatureCollection(config.get("CORE", "indiaMainlandID")).first()).simplify(500)
            ee.batch.Export.image.toAsset(** {
              'image': zones,
              'description': assetName,
              'assetId': assetFolder + assetName,
              'scale': 30,
              'pyramidingPolicy': {'.default': 'mode'},
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return zones

def classificationZonesFromStatesOneHotEncoded(returnExisting = False, startFreshExport = False):
    def assignNumberToStateGroupZone(i):
        # Have to do -1 on i each time, because zoneNumberSeq starts from 1
        indexOfiIntoLists = ee.Number(i).subtract(1)
        zoneOfi = ee.Feature(allStates.filter(ee.Filter.inList("state", ee.List(stateGroupingsForZones.get(indexOfiIntoLists)))).union().first())
        return zoneOfi.set("zoneNum", zoneNumberSeq.getNumber(indexOfiIntoLists), "zoneLabel", zoneLabels.getString(indexOfiIntoLists))

    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    allStates = ee.FeatureCollection(config.get("CORE", "indiaStates"))
    configZonesStates = config["AOI-CLASSIFICATION-ZONES-STATES"]
    assetName   = configZonesStates.get("existingZonesOneHotEncoded")
    stateGroupingsForZones = ee.List(ast.literal_eval(configZonesStates.get("groupsOfStates")))
    zoneLabels             = ee.List(ast.literal_eval(configZonesStates.get("groupsOfStatesLabels")))
    zoneBandnamePrefix = configZonesStates.get("featureBandNamePrefix")

    if returnExisting == True:
        zones = ee.Image(assetFolder + assetName)
    else:
        zoneNumberSeq = ee.List.sequence(1, zoneLabels.length());
        statesgroupsWithZoneNumberAssigned = ee.FeatureCollection(zoneNumberSeq.map(assignNumberToStateGroupZone))
        statesZonesOhe = classificationZonesFromRegionsFeatureCollection(statesgroupsWithZoneNumberAssigned, zoneBandnamePrefix, oneHotEncoded = True)
        zones = statesZonesOhe

        if startFreshExport == True:
            reg = ee.Feature(ee.FeatureCollection(config.get("CORE", "indiaMainlandID")).first()).simplify(500)
            ee.batch.Export.image.toAsset(** {
              'image': zones,
              'description': assetName,
              'assetId': assetFolder + assetName,
              'scale': 30,
              'pyramidingPolicy': {'.default': 'mode'},
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return zones

def classificationZonesFromBiomesNumeric(returnExisting = False, startFreshExport = False):
    def assignNumberToBiomeZone(i):
        # Have to do -1 on i each time, because zoneNumberSeq starts from 1
        indexOfiIntoLists = ee.Number(i).subtract(1)
        zoneOfi = ee.Feature(allBiomesFC.filter(ee.Filter.eq("BIOME_NAME", biomeNames.get(indexOfiIntoLists))).union().first())
        return zoneOfi.set("zoneNum", zoneNumberSeq.getNumber(indexOfiIntoLists), "zoneLabel", biomeLabels.getString(indexOfiIntoLists))

    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    configZonesBiomes = config["AOI-CLASSIFICATION-ZONES-BIOMES"]
    assetName = configZonesBiomes.get("existingZonesNumeric")
    allBiomesFC = ee.FeatureCollection(configZonesBiomes.get("indiaBiomesAndEcoregions"))
    zoneBandnamePrefix = configZonesBiomes.get("featureBandNamePrefix")
    biomeNames = ee.List(ast.literal_eval(configZonesBiomes.get("biomeNames")))
    biomeLabels = ee.List(ast.literal_eval(configZonesBiomes.get("biomeLabels")))

    if returnExisting == True:
        zones = ee.Image(assetFolder + assetName)
    else:
        # Group states into zones, assign a numeric and a string label to
        zoneNumberSeq = ee.List.sequence(1, biomeLabels.length());
        biomesWithZoneNumberAssigned = ee.FeatureCollection(zoneNumberSeq.map(assignNumberToBiomeZone))
        biomeZonesNum = classificationZonesFromRegionsFeatureCollection(biomesWithZoneNumberAssigned, zoneBandnamePrefix, oneHotEncoded = False)
        zones = biomeZonesNum

        if startFreshExport == True:
            reg = ee.Feature(ee.FeatureCollection(config.get("CORE", "indiaMainlandID")).first()).simplify(500)
            ee.batch.Export.image.toAsset(** {
              'image': zones,
              'description': assetName,
              'assetId': assetFolder + assetName,
              'scale': 30,
              'pyramidingPolicy': {'.default': 'mode'},
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return zones

def classificationZonesFromBiomesOneHotEncoded(returnExisting = False, startFreshExport = False):
    def assignNumberToBiomeZone(i):
        # Have to do -1 on i each time, because zoneNumberSeq starts from 1
        indexOfiIntoLists = ee.Number(i).subtract(1)
        zoneOfi = ee.Feature(allBiomesFC.filter(ee.Filter.eq("BIOME_NAME", biomeNames.get(indexOfiIntoLists))).union().first())
        return zoneOfi.set("zoneNum", zoneNumberSeq.getNumber(indexOfiIntoLists), "zoneLabel", biomeLabels.getString(indexOfiIntoLists))

    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    configZonesBiomes = config["AOI-CLASSIFICATION-ZONES-BIOMES"]
    assetName = configZonesBiomes.get("existingZonesOneHotEncoded")
    allBiomesFC = ee.FeatureCollection(configZonesBiomes.get("indiaBiomesAndEcoregions"))
    zoneBandnamePrefix = configZonesBiomes.get("featureBandNamePrefix")
    biomeNames = ee.List(ast.literal_eval(configZonesBiomes.get("biomeNames")))
    biomeLabels = ee.List(ast.literal_eval(configZonesBiomes.get("biomeLabels")))

    if returnExisting == True:
        zones = ee.Image(assetFolder + assetName)
    else:
        zoneNumberSeq = ee.List.sequence(1, biomeLabels.length());
        biomesWithZoneNumberAssigned = ee.FeatureCollection(zoneNumberSeq.map(assignNumberToBiomeZone))
        biomeZonesOhe = classificationZonesFromRegionsFeatureCollection(biomesWithZoneNumberAssigned, zoneBandnamePrefix, oneHotEncoded = True)
        zones = biomeZonesOhe

        if startFreshExport == True:
            reg = ee.Feature(ee.FeatureCollection(config.get("CORE", "indiaMainlandID")).first()).simplify(500)
            ee.batch.Export.image.toAsset(** {
              'image': zones,
              'description': assetName,
              'assetId': assetFolder + assetName,
              'scale': 30,
              'pyramidingPolicy': {'.default': 'mode'},
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return zones

def classificationZonesFromGeologicalAgeNumeric(returnExisting = False, startFreshExport = False):
    def assignNumberToGeologicalAgeZone(i):
        # Have to do -1 on i each time, because zoneNumberSeq starts from 1
        indexOfiIntoLists = ee.Number(i).subtract(1)
        zoneOfi = ee.Feature(allGeologicalAgesFC.filter(ee.Filter.eq("GLG", geologicalAgeNames.get(indexOfiIntoLists))).first())
        return zoneOfi.set("zoneNum", zoneNumberSeq.getNumber(indexOfiIntoLists), "zoneLabel", geologicalAgeLabels.getString(indexOfiIntoLists))

    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    configZonesGeol = config["AOI-CLASSIFICATION-ZONES-GEOLOGICAL-AGE"]
    assetName = configZonesGeol.get("existingZonesNumeric")
    allGeologicalAgesFC = ee.FeatureCollection(configZonesGeol.get("indiaGeologicalAge"))
    zoneBandnamePrefix = configZonesGeol.get("featureBandNamePrefix")
    geologicalAgeNames = ee.List(ast.literal_eval(configZonesGeol.get("geologicalAgeNames")))
    geologicalAgeLabels = ee.List(ast.literal_eval(configZonesGeol.get("geologicalAgeLabels")))

    if returnExisting == True:
        zones = ee.Image(assetFolder + assetName)
    else:
        # Group states into zones, assign a numeric and a string label to
        zoneNumberSeq = ee.List.sequence(1, geologicalAgeLabels.length());
        geolWithZoneNumberAssigned = ee.FeatureCollection(zoneNumberSeq.map(assignNumberToGeologicalAgeZone))
        geolZonesNum = classificationZonesFromRegionsFeatureCollection(geolWithZoneNumberAssigned, zoneBandnamePrefix, oneHotEncoded = False)
        zones = geolZonesNum

        if startFreshExport == True:
            reg = ee.Feature(ee.FeatureCollection(config.get("CORE", "indiaMainlandID")).first()).simplify(500)
            ee.batch.Export.image.toAsset(** {
              'image': zones,
              'description': assetName,
              'assetId': assetFolder + assetName,
              'scale': 30,
              'pyramidingPolicy': {'.default': 'mode'},
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return zones

def classificationZonesFromGeologicalAgeOneHotEncoded(returnExisting = False, startFreshExport = False):
    def assignNumberToGeologicalAgeZone(i):
        # Have to do -1 on i each time, because zoneNumberSeq starts from 1
        indexOfiIntoLists = ee.Number(i).subtract(1)
        zoneOfi = ee.Feature(allGeologicalAgesFC.filter(ee.Filter.eq("GLG", geologicalAgeNames.get(indexOfiIntoLists))).first())
        return zoneOfi.set("zoneNum", zoneNumberSeq.getNumber(indexOfiIntoLists), "zoneLabel", geologicalAgeLabels.getString(indexOfiIntoLists))

    with open("config.ini", 'r') as f:
        fileContents = f.read()
    config = cp.RawConfigParser(allow_no_value = True, interpolation = cp.ExtendedInterpolation())
    config.read_file(io.StringIO(fileContents))

    assetFolder = config.get('CORE', 'assetFolderWithFeatures')
    configZonesGeol = config["AOI-CLASSIFICATION-ZONES-GEOLOGICAL-AGE"]
    assetName = configZonesGeol.get("existingZonesOneHotEncoded")
    allGeologicalAgesFC = ee.FeatureCollection(configZonesGeol.get("indiaGeologicalAge"))
    zoneBandnamePrefix = configZonesGeol.get("featureBandNamePrefix")
    geologicalAgeNames = ee.List(ast.literal_eval(configZonesGeol.get("geologicalAgeNames")))
    geologicalAgeLabels = ee.List(ast.literal_eval(configZonesGeol.get("geologicalAgeLabels")))

    if returnExisting == True:
        zones = ee.Image(assetFolder + assetName)
    else:
        zoneNumberSeq = ee.List.sequence(1, geologicalAgeLabels.length());
        geolWithZoneNumberAssigned = ee.FeatureCollection(zoneNumberSeq.map(assignNumberToGeologicalAgeZone))
        geolZonesOhe = classificationZonesFromRegionsFeatureCollection(geolWithZoneNumberAssigned, zoneBandnamePrefix, oneHotEncoded = True)
        zones = geolZonesOhe

        if startFreshExport == True:
            reg = ee.Feature(ee.FeatureCollection(config.get("CORE", "indiaMainlandID")).first()).simplify(500)
            ee.batch.Export.image.toAsset(** {
              'image': zones,
              'description': assetName,
              'assetId': assetFolder + assetName,
              'scale': 30,
              'pyramidingPolicy': {'.default': 'mode'},
              'region': reg.geometry(),
              'maxPixels': 1e13
            }).start()

    return zones
