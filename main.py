import streamlit as st
import geopandas as gpd
import pydeck as pdk
from shapely.geometry import Point
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import osmnx as ox
import pandas as pd


c1, c2 = st.columns(2)
with c1:
    latitude = st.number_input('Latitude', value=59.92, step=0.01)  # Example latitude
with c2:
    longitude = st.number_input('Longitude', value=10.73, step=0.01)  # Example longitude
radius_km = 1.5  # Radius in kilometers
center_point = Point(longitude, latitude)

# bygning
gdf_buildings = ox.geometries_from_point((latitude, longitude), tags={'building': True})
gdf_buildings = gdf_buildings.to_crs(epsg=4326)
gdf_buildings["coordinates"] = gdf_buildings["geometry"].apply(lambda geom: [list(geom.exterior.coords)] if geom else None)
if 'height' not in gdf_buildings.columns:
    gdf_buildings['height'] = None
try:
    gdf_buildings['height'] = gdf_buildings['height'].str.replace(' m', '', regex=False)  # Remove ' m'
except:
    pass 
gdf_buildings['height'] = pd.to_numeric(gdf_buildings['height'], errors='coerce')  # Convert to numeric
gdf_buildings['height'].fillna(10, inplace=True)
center_point = gpd.GeoSeries([center_point], crs="EPSG:4326").to_crs(gdf_buildings.crs).iloc[0]
circle = center_point.buffer(0.005)  # Convert km to meters
gdf_buildings = gdf_buildings[gdf_buildings.intersects(circle)]

# løsmasser
gdf = gpd.read_file('zip://løsmasseFlate.zip')
gdf = gdf.to_crs(epsg=32632)  # Replace with correct UTM zone for your area

center_point = gpd.GeoSeries([center_point], crs="EPSG:4326").to_crs(gdf.crs).iloc[0]
circle = center_point.buffer(radius_km * 1000)  # Convert km to meters
gdf = gdf[gdf.intersects(circle)]

gdf['geometry'] = gdf.intersection(circle)
gdf = gdf.explode(index_parts=False)
gdf["coordinates"] = gdf["geometry"].apply(lambda geom: [list(geom.exterior.coords)] if geom else None)

gdf = gdf[['jorda_navn', 'geometry']]
gdf = gdf.explode(index_parts=False)

unique_values = list(gdf['jorda_navn'].unique())

custom_colors = {
    "Forvitringsmateriale, ikke inndelt etter mektighet": [232,194,255,100],  
    "Fyllmasse (antropogent materiale)": [174,174,174,100], 
    "Hav- og fjordavsetning, sammenhengende dekke, stedvis med stor mektighet": [64,191,255,100],
    "Torv og myr": [196,148,126,100],
    "Elve- og bekkeavsetning (Fluvial avsetning)": [255,237,97,100],
}

def assign_color(value):
    return custom_colors.get(value, [128, 128, 128, 255])  # Default to gray if not found

gdf["color"] = gdf['jorda_navn'].apply(assign_color)
gdf = gdf.to_crs(epsg=4326)

gdf["coordinates"] = gdf["geometry"].apply(lambda geom: [list(geom.exterior.coords)] if geom else None)
df = gdf[["coordinates", "jorda_navn", "color"]]

building_layer = pdk.Layer(
        "PolygonLayer",
        gdf_buildings,
        get_polygon="coordinates",
        get_fill_color=[160, 160, 160, 155],  # Color for buildings
        get_elevation="height",  # Use height data for 3D extrusion
        extruded=True,
        pickable=True,
        elevation_scale=1,  # Adjust if necessary for better visualization
        wireframe=True,
)
# Define the Pydeck layer
polygon_layer = pdk.Layer(
    "PolygonLayer",
    data=df,
    get_polygon="coordinates",
    get_fill_color="color",  # Customize the color
    pickable=True,
    auto_highlight=True,
    tooltip={"text": "{jorda_navn}"}
)

point_layer = pdk.Layer(
    "ScatterplotLayer",
    data=[{"position": [longitude, latitude]}],
    get_position="position",
    get_color=[255, 0, 0, 100],  
    get_radius=100,  # Adjust the radius for visibility
)

# Define the view state for the map
view_state = pdk.ViewState(
    latitude=latitude,
    longitude=longitude,
    zoom=12,
    pitch=60
)

# Render in Streamlit
st.pydeck_chart(pdk.Deck(
    layers=[polygon_layer, building_layer, point_layer],
    initial_view_state=view_state,
    tooltip={"text": "{jorda_navn}"},
    map_style='mapbox://styles/mapbox/light-v10',
), height=400)

#-- ML

def knn_geology(polygons, point, k=5):
    # Create a feature matrix and labels
    features = []
    labels = []
    
    for _, row in polygons.iterrows():
        polygon_centroid = row.geometry.centroid
        features.append([polygon_centroid.x, polygon_centroid.y])
        labels.append(row['jorda_navn'])

    # Fit KNN
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels)

    # Predict for the point
    point_prediction = knn.predict([[point.x, point.y]])
    return point_prediction[0]

# Determine likely geology using KNN
likely_geology = knn_geology(gdf, center_point)
st.write(f"The likely geology at the point ({latitude}, {longitude}) using KNN is: {likely_geology}")
