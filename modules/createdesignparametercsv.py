import csv
from zipfile import ZipFile
from pykml import parser


class CSVCreator:
    def __init__(self, file, fields=None):
        self.file = file
        self.fields = fields

    def add_entries(self, gcp_dict):
        with open(self.file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fields)
            writer.writeheader()
            for image_name, contour_results in gcp_dict.items():
                contour = 1
                for cont_number, result in contour_results.items():
                    writer.writerow({self.fields[0]: image_name,
                                     self.fields[1]: contour,
                                     self.fields[2]:
                                         result["SideLengthsMinBoundRect"][0],
                                     self.fields[3]:
                                         result["SideLengthsMinBoundRect"][1],
                                     self.fields[4]: result[
                                         "HeightWidthRatioMinBoundRect"],
                                     self.fields[5]: result[
                                         "AreaRatioContourMinBoundRect"],
                                     self.fields[6]: result[
                                         "PerimeterRatioContourMinBoundRect"],
                                     self.fields[7]: result[
                                         "ActualGCPAreaRatio"],
                                     self.fields[8]: result[
                                         "ActualGCPPerimeterRatio"],
                                     self.fields[9]: result[
                                         "RatioDefectLineLength"],
                                     self.fields[10]: result["GCPLocation"][0],
                                     self.fields[11]: result["GCPLocation"][1]
                                     })
                contour += 1

    # def keyholemarkup2x(self, output):
    #     """
    #     Takes Keyhole Markup Language Zipped (KMZ) or KML file as input. The
    #     output is a pandas dataframe, geopandas geodataframe, csv, geojson, or
    #     shapefile.
    #
    #     All core functionality from:
    #     http://programmingadvent.blogspot.com/2013/06/kmzkml-file-parsing-with-python.html
    #
    #     Parameters
    #         ----------
    #         file : {string}
    #             The string path to your KMZ or .
    #         output : {string}
    #             Defines the type of output. Valid selections include:
    #                 - shapefile - 'shp', 'shapefile', or 'ESRI Shapefile'
    #
    #         Returns
    #         -------
    #         self : object
    #     """
    #     r = re.compile(r'(?<=\.)km+[lz]?', re.I)
    #     try:
    #         extension = r.search(self.file).group(0)  # (re.findall(r'(?<=\.)[\w]+',file))[-1]
    #
    #
    #     except IOError as e:
    #         raise IOError("I/O error {0}".format(e))
    #     if (extension.lower() == 'kml') is True:
    #         buffer = self.file
    #     elif (extension.lower() == 'kmz') is True:
    #         kmz = ZipFile(self.file, 'r')
    #
    #         vmatch = np.vectorize(lambda x: bool(r.search(x)))
    #         A = np.array(kmz.namelist())
    #         sel = vmatch(A)
    #         buffer = kmz.open(A[sel][0], 'r')
    #
    #     else:
    #         raise ValueError(
    #             'Incorrect file format entered.  Please provide the '
    #             'path to a valid KML or KMZ file.')
    #
    #     parser = xml.sax.make_parser()
    #     handler = PlacemarkHandler()
    #     parser.setContentHandler(handler)
    #     parser.parse(buffer)
    #
    #     try:
    #         kmz.close()
    #     except:
    #         pass
    #
    #     df = pd.DataFrame(handler.mapping).T
    #     names = list(map(lambda x: x.lower(), df.columns))
    #     if 'description' in names:
    #         extradata = df.apply(PlacemarkHandler.htmlizer, axis=1)
    #         df = df.join(extradata)
    #
    #     output = output.lower()
    #
    #     if output == 'df' or output == 'dataframe' or output == None:
    #         result = df
    #
    #     elif output == 'csv':
    #         out_filename = self.file[:-3] + "csv"
    #         df.to_csv(out_filename, encoding='utf-8', sep="\t")
    #         result = ("Successfully converted {0} to CSV and output to"
    #                   " disk at {1}".format(self.file, out_filename))
    #
    #     elif output == 'gpd' or output == 'gdf' or output == 'geoframe' or output == 'geodataframe':
    #         try:
    #             import shapely
    #             from shapely.geometry import Polygon, LineString, Point
    #         except ImportError as e:
    #             raise ImportError(
    #                 'This operation requires shapely. {0}'.format(e))
    #         try:
    #             import fiona
    #         except ImportError as e:
    #             raise ImportError(
    #                 'This operation requires fiona. {0}'.format(e))
    #         try:
    #             import geopandas as gpd
    #         except ImportError as e:
    #             raise ImportError(
    #                 'This operation requires geopandas. {0}'.format(e))
    #
    #         geos = gpd.GeoDataFrame(
    #             df.apply(PlacemarkHandler.spatializer, axis=1))
    #         result = gpd.GeoDataFrame(pd.concat([df, geos], axis=1))
    #
    #
    #     elif output == 'geojson' or output == 'json':
    #         try:
    #             import shapely
    #             from shapely.geometry import Polygon, LineString, Point
    #         except ImportError as e:
    #             raise ImportError(
    #                 'This operation requires shapely. {0}'.format(e))
    #         try:
    #             import fiona
    #         except ImportError as e:
    #             raise ImportError(
    #                 'This operation requires fiona. {0}'.format(e))
    #         try:
    #             import geopandas as gpd
    #         except ImportError as e:
    #             raise ImportError(
    #                 'This operation requires geopandas. {0}'.format(e))
    #         try:
    #             import geojson
    #         except ImportError as e:
    #             raise ImportError(
    #                 'This operation requires geojson. {0}'.format(e))
    #
    #         geos = gpd.GeoDataFrame(
    #             df.apply(PlacemarkHandler.spatializer, axis=1))
    #         gdf = gpd.GeoDataFrame(pd.concat([df, geos], axis=1))
    #         out_filename = self.file[:-3] + "geojson"
    #         gdf.to_file(out_filename, driver='GeoJSON')
    #         validation = geojson.is_valid(geojson.load(open(out_filename)))[
    #             'valid']
    #         if validation == 'yes':
    #
    #             result = ("Successfully converted {0} to GeoJSON and output to"
    #                       " disk at {1}".format(self.file, out_filename))
    #         else:
    #             raise ValueError('The geojson conversion did not create a '
    #                              'valid geojson object. Try to clean your '
    #                              'data or try another file.')
    #
    #     elif output == 'shapefile' or output == 'shp' or output == 'esri shapefile':
    #         try:
    #             import shapely
    #             from shapely.geometry import Polygon, LineString, Point
    #         except ImportError as e:
    #             raise ImportError(
    #                 'This operation requires shapely. {0}'.format(e))
    #         try:
    #             import fiona
    #         except ImportError as e:
    #             raise ImportError(
    #                 'This operation requires fiona. {0}'.format(e))
    #
    #         try:
    #             import geopandas as gpd
    #         except ImportError as e:
    #             raise ImportError(
    #                 'This operation requires geopandas. {0}'.format(e))
    #
    #         try:
    #             import shapefile
    #         except ImportError as e:
    #             raise ImportError(
    #                 'This operation requires pyshp. {0}'.format(e))
    #
    #         geos = gpd.GeoDataFrame(
    #             df.apply(PlacemarkHandler.spatializer, axis=1))
    #         gdf = gpd.GeoDataFrame(pd.concat([df, geos], axis=1))
    #         out_filename = self.file[:-3] + "shp"
    #         gdf.to_file(out_filename, driver='ESRI Shapefile')
    #         sf = shapefile.Reader(out_filename)
    #         import shapefile
    #         sf = shapefile.Reader(out_filename)
    #         if len(sf.shapes()) > 0:
    #             validation = "yes"
    #         else:
    #             validation = "no"
    #         if validation == 'yes':
    #
    #             result = (
    #                 "Successfully converted {0} to Shapefile and output to"
    #                 " disk at {1}".format(self.file, out_filename))
    #         else:
    #             raise ValueError('The Shapefile conversion did not create a '
    #                              'valid shapefile object. Try to clean your '
    #                              'data or try another file.')
    #     else:
    #         raise ValueError('The conversion returned no data; check if'
    #                          ' you entered a correct output file type. '
    #                          'Valid output types are geojson, shapefile,'
    #                          ' csv, geodataframe, and/or pandas dataframe.')
    #
    #     return result

    def convert_kmz_to_kml(self):
        kmz = ZipFile(self.file, 'r')
        with open('doc.kml', 'r') as kml_file:
            doc = parser.parse(kml_file)
            print(doc)


