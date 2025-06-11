import dash
from dash import dcc, html, Input, Output, State, no_update, Patch
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

import numpy as np
import scipy.io
from skimage.measure import marching_cubes
from stl import mesh as np_stl_mesh
import logging
import base64
import io
from datetime import datetime
import os
import tempfile
import json
import time # For timing operations
import sys # For checking object sizes

# --- Constants & Configuration ---
APP_TITLE = "MRI Voxel Visualizer with Three.js (Downsampling)"
LOG_FILENAME = "mri_processing_log_threejs_ds.txt"
PROCESSED_DATA_SUFFIX = "_processed.npy"
DEFAULT_THRESHOLD = 0.1
DEFAULT_SPACING = 1.0
DEFAULT_POINT_SIZE = 1.0
DEFAULT_DOWNSAMPLE_FACTOR = 1 # 1 means no downsampling

# --- Logging Setup ---
# (Remove old handlers if re-running in same session, to avoid duplicate log entries)
root_logger = logging.getLogger()
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILENAME, mode='a'),
        logging.StreamHandler()
    ]
)
def log_action(message):
    logging.info(message)

# --- Helper Functions ---
def generate_dummy_mat_file(filename="dummy_mri_threejs_ds.mat", var_name="mri_data_var"):
    # (Same as before)
    if not os.path.exists(filename):
        size = 30; x_idx,y_idx,z_idx=np.indices((size,size,size)); data=((x_idx-size/2)**2+(y_idx-size/2)**2+(z_idx-size/2)**2<(size/3)**2).astype(np.float32); data+=np.random.rand(size,size,size)*0.3*data; data=np.clip(data,0,1.5); scipy.io.savemat(filename,{var_name:data}); log_action(f"Generated dummy MAT: {filename} key '{var_name}'")
DUMMY_MAT_VAR_NAME = "mri_data_ds"; generate_dummy_mat_file(var_name=DUMMY_MAT_VAR_NAME)

def parse_contents(contents, filename, mat_var_name=None):
    # (Same as before, ensure logging is good)
    t_start = time.time()
    log_action(f"Parsing file: {filename} (MAT var hint: {mat_var_name})")
    content_type, content_string = contents.split(','); decoded = base64.b64decode(content_string)
    data_array, original_fn_base, error_msg, metadata = None, filename, None, {'filename': filename}
    try:
        if filename.endswith('.mat'):
            mat_data = scipy.io.loadmat(io.BytesIO(decoded)); found_key = None
            if mat_var_name and mat_var_name in mat_data:
                value = mat_data[mat_var_name]
                if isinstance(value,np.ndarray) and value.ndim==3: data_array=value.astype(np.float32); found_key=mat_var_name
                else: error_msg=f"Key '{mat_var_name}' not 3D array."
            if data_array is None and error_msg is None:
                for k,v in mat_data.items():
                    if k.startswith('__'): continue
                    if isinstance(v,np.ndarray) and v.ndim==3: data_array=v.astype(np.float32); found_key=k; break
            if data_array is not None: metadata.update({'dimensions':data_array.shape, 'original_min':float(np.min(data_array)), 'original_max':float(np.max(data_array)), 'mat_key_used': found_key})
            elif error_msg is None: error_msg="No 3D numpy array in .mat."
        elif filename.endswith(PROCESSED_DATA_SUFFIX) or filename.endswith('.npy'):
            data_array = np.load(io.BytesIO(decoded)).astype(np.float32); metadata.update({'dimensions':data_array.shape, 'is_pre_normalized':True})
            if filename.endswith(PROCESSED_DATA_SUFFIX): original_fn_base=filename.replace(PROCESSED_DATA_SUFFIX,".mat")
        else: error_msg="Unsupported file type."
        log_action(f"Parsing done in {time.time()-t_start:.3f}s. Result: {'Success' if data_array is not None else 'Fail ('+str(error_msg)+')'}")
        return (data_array, original_fn_base, metadata, error_msg) if error_msg or data_array is not None else (None, original_fn_base, metadata, "Unknown parsing issue.")
    except Exception as e: log_action(f"EXCEPTION during parsing: {e}"); return None,filename,{},f"Error: {e}"

def normalize_data(data_array):
    # (Same as before)
    t_start = time.time()
    if data_array is None or data_array.size==0: return None
    min_v,max_v = np.min(data_array),np.max(data_array)
    norm = (data_array-min_v)/(max_v-min_v) if max_v > min_v else np.zeros_like(data_array)
    log_action(f"Normalization done in {time.time()-t_start:.3f}s. Range [{min_v:.2f},{max_v:.2f}]->[0,1]")
    return norm.astype(np.float32)

def filter_voxels_for_threejs(normalized_data, threshold, invert_cutoff, downsample_factor):
    t_start = time.time()
    if normalized_data is None: return []
    
    if invert_cutoff: indices = np.argwhere(normalized_data <= threshold)
    else: indices = np.argwhere(normalized_data > threshold)
    log_action(f"np.argwhere took {time.time()-t_start:.3f}s. Found {indices.shape[0]} raw points.")

    if indices.size == 0:
        log_action(f"No voxels {'below' if invert_cutoff else 'above'} threshold {threshold:.2f}.")
        return []

    # Apply downsampling
    if downsample_factor > 1 and indices.shape[0] > 0:
        t_ds_start = time.time()
        num_to_keep = indices.shape[0] // downsample_factor
        if num_to_keep == 0 and indices.shape[0] > 0: # Ensure at least one point if possible
            num_to_keep = 1 
        
        # Random sampling is often good for visualization to avoid aliasing from slicing
        # If you need deterministic downsampling, use slicing: indices[::downsample_factor]
        rng = np.random.default_rng()
        sampled_indices_idx = rng.choice(indices.shape[0], size=num_to_keep, replace=False)
        indices = indices[sampled_indices_idx]
        log_action(f"Downsampling (factor {downsample_factor}) from {indices.shape[0]*downsample_factor if num_to_keep > 0 else 'many'} to {indices.shape[0]} points took {time.time()-t_ds_start:.3f}s.")
    elif downsample_factor <= 1:
        log_action(f"No downsampling applied (factor {downsample_factor}).")


    i_coords, j_coords, k_coords = indices[:,0], indices[:,1], indices[:,2]
    values = normalized_data[i_coords, j_coords, k_coords]
    
    t_list_start = time.time()
    threejs_points = [[int(i),int(j),int(k),float(v)] for i,j,k,v in zip(i_coords,j_coords,k_coords,values)]
    log_action(f"List construction for {len(threejs_points)} points took {time.time()-t_list_start:.3f}s.")
    log_action(f"Total filter_voxels took {time.time()-t_start:.3f}s. Output {len(threejs_points)} points.")
    return threejs_points

def generate_stl_from_voxels(voxel_data, threshold_value, spacing):
    # (Same as before, ensure logging for timing)
    t_start = time.time()
    # ... (rest of STL generation logic) ...
    log_action(f"STL generation took {time.time()-t_start:.3f}s.")
    # ... (return stl_bytes, message) ...
    if voxel_data is None: return None, "Error: No voxel data for STL."
    try:
        verts,faces,_,_ = marching_cubes(volume=voxel_data,level=threshold_value,spacing=spacing,method='lewiner')
        if verts.size==0 or faces.size==0: return None, "Marching cubes: empty mesh."
        mesh=np_stl_mesh.Mesh(np.zeros(faces.shape[0],dtype=np_stl_mesh.Mesh.dtype)); mesh.vectors=verts[faces]
        with tempfile.NamedTemporaryFile(suffix=".stl",delete=False) as tmp_f: mesh.save(tmp_f.name); tmp_fname=tmp_f.name
        with open(tmp_fname,'rb') as f: stl_bytes=f.read()
        os.remove(tmp_fname)
        log_action(f"STL gen done in {time.time()-t_start:.3f}s. {len(faces)} faces.")
        return stl_bytes, f"STL: {len(faces)} faces."
    except Exception as e: log_action(f"STL EXCEPTION: {e}"); return None, f"STL Error: {e}"


# --- Dash Application ---
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                external_scripts=[
                    "https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js",
                    "https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"
                ],
                title=APP_TITLE)
server = app.server
THREEJS_COLORMAPS = ['rainbow', 'grayscale', 'heatmap']

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1(APP_TITLE), width=12, className="mb-4 mt-4 text-center")),
    dcc.Store(id='store-normalized-data-array', storage_type='memory'),
    dcc.Store(id='store-data-metadata', storage_type='memory'),
    dcc.Store(id='store-current-filename-base', storage_type='session'),
    dcc.Store(id='store-threejs-render-trigger'),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Controls & Upload"),
                dbc.CardBody([
                    # ... (Upload, MAT var name - same as before) ...
                    dcc.Upload(id='upload-data', children=html.Div(['Drag & Drop or ', html.A('Select .mat/.npy')]), style={'width':'100%','height':'60px','lineHeight':'60px','borderWidth':'1px','borderStyle':'dashed','borderRadius':'5px','textAlign':'center','margin':'10px 0'}),
                    dbc.Input(id="mat-var-name-input", placeholder="MAT Var Name (opt.)", type="text", className="mb-2"),
                    html.Div(id='output-file-info', className="mt-2 mb-2 small"),
                    dbc.Spinner(html.Div(id='processing-status', className="small"), color="primary"),
                    html.Hr(),

                    html.H5("Voxel Spacing", className="mt-3"),
                    # ... (Spacing inputs - same as before) ...
                    dbc.Row([dbc.Col(dbc.Input(id="spacing-x", placeholder="X",type="number",value=DEFAULT_SPACING,min=0.01,step=0.01),md=4), dbc.Col(dbc.Input(id="spacing-y",placeholder="Y",type="number",value=DEFAULT_SPACING,min=0.01,step=0.01),md=4), dbc.Col(dbc.Input(id="spacing-z",placeholder="Z",type="number",value=DEFAULT_SPACING,min=0.01,step=0.01),md=4)], className="mb-2"),

                    html.H5("Filtering & Visualization", className="mt-3"),
                    # ... (Threshold, Invert, Point Size, Colormap - same as before) ...
                    dbc.Label("Threshold Level:",html_for="threshold-slider"), dcc.Slider(id='threshold-slider',min=0,max=1,step=0.01,value=DEFAULT_THRESHOLD,marks={i/10:str(round(i/10,1)) for i in range(11)},tooltip={"placement":"bottom","always_visible":True},disabled=True),
                    dbc.Checkbox(id="invert-cutoff-checkbox",label="Invert Cutoff",value=False,className="mt-1 mb-2",disabled=True),
                    
                    dbc.Label("Downsample Factor (for Viz):", html_for="downsample-slider"),
                    dcc.Slider(id='downsample-slider', min=1, max=100, step=1, value=DEFAULT_DOWNSAMPLE_FACTOR,
                               marks={1: '1x', 10: '10x', 20: '20x', 50: '50x', 100: '100x (fewest points)'},
                               tooltip={"placement": "bottom", "always_visible": True}, disabled=True),

                    dbc.Label("Point Size:",html_for="point-size-slider", className="mt-2"), dcc.Slider(id='point-size-slider',min=0.1,max=5,step=0.1,value=DEFAULT_POINT_SIZE,marks={i:str(i) for i in range(6)},tooltip={"placement":"bottom","always_visible":True},disabled=True),
                    dbc.Label("Colormap:",html_for="colormap-dropdown",className="mt-3"), dcc.Dropdown(id='colormap-dropdown',options=[{'label':cm.capitalize(),'value':cm} for cm in THREEJS_COLORMAPS],value=THREEJS_COLORMAPS[0],clearable=False,disabled=True),
                    html.Hr(),

                    html.H5("Export & Actions", className="mt-3"),
                    # ... (Buttons - same as before) ...
                    dbc.Button("Save Processed (.npy)",id="btn-save-processed",color="success",className="me-2 mt-2",disabled=True,n_clicks=0), dbc.Button("Generate & Download STL",id="btn-generate-stl",color="primary",className="me-2 mt-2",disabled=True,n_clicks=0), html.Div(id='stl-status-message',className="mt-2 small"),
                    dbc.Button("Download Log",id="btn-download-log",color="info",className="me-2 mt-2",n_clicks=0), dbc.Button("Reset UI",id="btn-reset-data",color="warning",className="mt-2",n_clicks=0), html.Br(),
                    dbc.Button("Toggle Rotation (3JS)",id="btn-threejs-rotate",color="secondary",className="me-2 mt-2",disabled=True,n_clicks=0), dbc.Button("Reset View (3JS)",id="btn-threejs-reset-view",color="secondary",className="mt-2",disabled=True,n_clicks=0),
                    dcc.Download(id="download-processed-data"), dcc.Download(id="download-stl-file"), dcc.Download(id="download-log-file"),
                ])
            ])
        ], md=4),
        dbc.Col([
            # ... (Three.js canvas container - same as before) ...
            dbc.Card([dbc.CardHeader("3D Visualization (Three.js)"), dbc.CardBody([html.Div(id='threejs-canvas-container',style={'width':'100%','height':'70vh','backgroundColor':'#111111','position':'relative'},children=[html.Canvas(id='threejs-canvas',style={'width':'100%','height':'100%'}),html.Div(id='threejs-loading-indicator',children=[dbc.Spinner(color="light")," Processing..."],style={'position':'absolute','top':'50%','left':'50%','transform':'translate(-50%,-50%)','color':'white','display':'none','zIndex':'10'})])]), dbc.CardFooter(id="threejs-stats-footer",children="Stats will appear here.",className="small")])
        ], md=8)
    ]),
    dbc.Row(dbc.Col(html.P(f"Dummy MAT file '{DUMMY_MAT_VAR_NAME}' generated.", className="text-muted small"),width=12,className="text-center mt-3"))
], fluid=True)


# --- Callbacks ---
@app.callback(
    [Output('store-normalized-data-array', 'data'), Output('store-data-metadata', 'data'),
     Output('store-current-filename-base', 'data'), Output('output-file-info', 'children'),
     Output('processing-status', 'children'), Output('threshold-slider', 'disabled'),
     Output('invert-cutoff-checkbox', 'disabled'), Output('downsample-slider', 'disabled'), # New
     Output('point-size-slider', 'disabled'), Output('colormap-dropdown', 'disabled'),
     Output('btn-save-processed', 'disabled'), Output('btn-generate-stl', 'disabled'),
     Output('btn-threejs-rotate', 'disabled'), Output('btn-threejs-reset-view', 'disabled'),
     Output('stl-status-message', 'children', allow_duplicate=True)],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'), State('mat-var-name-input', 'value')],
    prevent_initial_call=True
)
def process_uploaded_file(contents, filename, mat_var_name):
    t_start_cb = time.time()
    log_action(f"CB process_uploaded_file: START for {filename}")
    if contents is None: raise PreventUpdate
    
    raw_data, base_fn, metadata, err = parse_contents(contents, filename, mat_var_name if mat_var_name and mat_var_name.strip() else None)

    num_disabled_outputs = 10 # threshold, invert, downsample, pointsize, colormap, save, stl, rotate, reset_view
    base_out = [no_update]*3 + [dbc.Alert(err,color="danger") if err else "",""] + [True]*num_disabled_outputs + [""]
    if err: log_action(f"CB process_uploaded_file: Error parsing - {err}"); return base_out
    if raw_data is None: log_action(f"CB process_uploaded_file: No data from parsing."); return base_out

    status = []
    if metadata.get('is_pre_normalized'):
        norm_data = raw_data; status.append(f"Loaded pre-processed '{filename}'.")
        min_v,max_v = np.min(norm_data),np.max(norm_data)
        if not ((abs(min_v)<0.001 and abs(max_v-1.0)<0.001) or (min_v==0 and max_v==0)):
            log_action(f"CB process_uploaded_file: .npy not normalized. Re-normalizing."); norm_data=normalize_data(raw_data); status.append("Re-normalized.")
    else: norm_data=normalize_data(raw_data); status.append(f"Loaded & normalized '{filename}'.")

    if norm_data is None: log_action(f"CB process_uploaded_file: Normalization failed."); return [no_update]*3 + [dbc.Alert("Norm failed.",color="danger"),""] + [True]*num_disabled_outputs + [""]

    metadata.update({'normalized_min':float(np.min(norm_data)), 'normalized_max':float(np.max(norm_data))})
    file_info = f"Loaded: {filename}. Base: {base_fn}. Shape: {metadata.get('dimensions')}"
    
    log_action(f"CB process_uploaded_file: END in {time.time()-t_start_cb:.3f}s.")
    return (norm_data.tolist(), metadata, os.path.splitext(base_fn)[0],
            dbc.Alert(file_info,color="success",className="small"), " ".join(status),
            False, False, False, False, False, False, False, False, False, "") # Enable controls


@app.callback(
    [Output('store-threejs-render-trigger', 'data'), Output('threejs-stats-footer', 'children'),
     Output('threejs-loading-indicator', 'style')],
    [Input('store-normalized-data-array', 'data'), Input('threshold-slider', 'value'),
     Input('invert-cutoff-checkbox', 'value'), Input('downsample-slider', 'value'), # New
     Input('point-size-slider', 'value'), Input('colormap-dropdown', 'value'),
     Input('spacing-x', 'value'), Input('spacing-y', 'value'), Input('spacing-z', 'value')],
    [State('store-data-metadata', 'data')]
)
def update_threejs_visualization_data(
    norm_data_json, threshold, invert_cutoff, downsample_factor, point_size, colormap,
    sx, sy, sz, data_metadata
):
    t_start_cb = time.time()
    log_action(f"CB update_threejs_viz_data: START. Downsample factor: {downsample_factor}")
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    if norm_data_json is None or data_metadata is None: return no_update, "Load data.", {'display':'none'}

    loader_show = {'position':'absolute','top':'50%','left':'50%','transform':'translate(-50%,-50%)','color':'white','zIndex':'10','display':'flex','alignItems':'center'}
    loader_hide = {'display':'none'}
    current_loader = loader_hide
    # Show loader for more intensive operations
    if trigger_id in ['store-normalized-data-array','threshold-slider','invert-cutoff-checkbox','downsample-slider','spacing-x','spacing-y','spacing-z']:
        current_loader = loader_show
    
    norm_data_arr = np.array(norm_data_json)
    log_action(f"CB update_threejs_viz_data: Converted norm_data_json to np.array in {time.time()-t_start_cb:.3f}s so far.")
    
    threejs_pts = filter_voxels_for_threejs(norm_data_arr, threshold, invert_cutoff, downsample_factor)
    
    viz_opts = {'threshold':threshold,'invertCutoff':invert_cutoff,'downsampleFactor':downsample_factor,
                'pointSize':point_size,'colorScheme':colormap,
                'spacingX':float(sx if sx is not None else DEFAULT_SPACING),
                'spacingY':float(sy if sy is not None else DEFAULT_SPACING),
                'spacingZ':float(sz if sz is not None else DEFAULT_SPACING)}
    
    stats = (f"File: {data_metadata.get('filename','N/A')}, Dims: {data_metadata.get('dimensions','N/A')}, "
             f"Displaying: {len(threejs_pts)} voxels (DS factor: {downsample_factor}x)")
    
    render_data = {'points':threejs_pts,'vizOptions':viz_opts,'metadata':data_metadata}
    
    # Approx size logging (can be slow for huge data, use sparingly or estimate)
    # list_size_mb = sys.getsizeof(threejs_pts) / (1024*1024)
    # json_size_mb = len(json.dumps(render_data)) / (1024*1024)
    # log_action(f"CB update_threejs_viz_data: Approx points list size: {list_size_mb:.2f}MB. Approx JSON to send: {json_size_mb:.2f}MB")
    log_action(f"CB update_threejs_viz_data: END in {time.time()-t_start_cb:.3f}s.")
    return render_data, stats, current_loader

# Client-side callback (same as before)
app.clientside_callback(
    """
    function(renderTriggerData, n_clicks_rotate, n_clicks_reset) {
        const ctx = dash_clientside.callback_context;
        // console.log("Clientside CB triggered by:", ctx.triggered.map(t => t.prop_id));
        if (!window.MRIViewer || typeof window.MRIViewer.updatePointCloud !== 'function') {
            // console.error("MRIViewer or methods not available yet in clientside_callback.");
            return dash_clientside.no_update;
        }
        if (ctx.triggered) {
            const triggered_id_full = ctx.triggered[0].prop_id;
            const triggered_id = triggered_id_full.split('.')[0];
            if (triggered_id === 'btn-threejs-rotate') window.MRIViewer.toggleRotation();
            else if (triggered_id === 'btn-threejs-reset-view') window.MRIViewer.resetView();
        }
        if (renderTriggerData && (ctx.triggered.some(t => t.prop_id === 'store-threejs-render-trigger.data'))) {
            if (renderTriggerData.points !== undefined && renderTriggerData.vizOptions && renderTriggerData.metadata) {
                // console.log("Clientside: Calling MRIViewer.updatePointCloud");
                window.MRIViewer.updatePointCloud(
                    renderTriggerData.points, renderTriggerData.vizOptions, renderTriggerData.metadata
                );
                setTimeout(() => { // Hide loader after JS processing (hopefully)
                    const loader = document.getElementById('threejs-loading-indicator');
                    if(loader) loader.style.display = 'none';
                }, 150);
            }
        }
        return dash_clientside.no_update;
    }
    """,
    Output('store-threejs-render-trigger', 'data', allow_duplicate=True), 
    Input('store-threejs-render-trigger', 'data'),
    Input('btn-threejs-rotate', 'n_clicks'),
    Input('btn-threejs-reset-view', 'n_clicks'),
    prevent_initial_call=True
)

# Callbacks for Save Processed, Generate STL, Download Log, Reset (largely same as before)
@app.callback(Output('download-processed-data','data'),[Input('btn-save-processed','n_clicks')],[State('store-normalized-data-array','data'),State('store-current-filename-base','data')],prevent_initial_call=True)
def save_data(n,data_json,base_fn):
    if data_json is None or base_fn is None: raise PreventUpdate
    data=np.array(data_json); out_fn=f"{base_fn}{PROCESSED_DATA_SUFFIX}"; buf=io.BytesIO(); np.save(buf,data); buf.seek(0)
    log_action(f"CB save_data: Prepared {out_fn}"); return dcc.send_bytes(buf.getvalue(),filename=out_fn)

@app.callback([Output('download-stl-file','data'),Output('stl-status-message','children')],[Input('btn-generate-stl','n_clicks')],[State('store-normalized-data-array','data'),State('threshold-slider','value'),State('store-current-filename-base','data'),State('spacing-x','value'),State('spacing-y','value'),State('spacing-z','value')],prevent_initial_call=True)
def gen_stl(n,data_json,thresh,base_fn,sx,sy,sz):
    log_action("CB gen_stl: START")
    if data_json is None or base_fn is None: return no_update,dbc.Alert("No data for STL.",color="warning",className="small")
    data=np.array(data_json); # Full resolution data used here
    spc_x,spc_y,spc_z = (float(s if s is not None else DEFAULT_SPACING) for s in (sx,sy,sz))
    out_fn=f"{base_fn}_thresh{thresh:.2f}_spc{spc_x:.1f}_{spc_y:.1f}_{spc_z:.1f}.stl"
    stl_bytes,msg = generate_stl_from_voxels(data,thresh,(spc_x,spc_y,spc_z))
    if stl_bytes: log_action(f"CB gen_stl: {out_fn} ready."); return dcc.send_bytes(stl_bytes,filename=out_fn),dbc.Alert(msg,color="success",className="small")
    else: log_action(f"CB gen_stl: Failed. {msg}"); return no_update,dbc.Alert(f"STL Fail: {msg}",color="danger",className="small")

@app.callback(Output('download-log-file','data'),[Input('btn-download-log','n_clicks')],prevent_initial_call=True)
def dl_log(n): log_action("Log download req."); [h.flush() for h in logging.getLogger().handlers]; return dcc.send_file(LOG_FILENAME)

@app.callback(
    [Output('store-normalized-data-array','clear_data'), Output('store-data-metadata','clear_data'),
     Output('store-current-filename-base','clear_data'), Output('store-threejs-render-trigger','clear_data',allow_duplicate=True),
     Output('output-file-info','children',allow_duplicate=True), Output('processing-status','children',allow_duplicate=True),
     Output('threshold-slider','disabled',allow_duplicate=True), Output('threshold-slider','value',allow_duplicate=True),
     Output('invert-cutoff-checkbox','disabled',allow_duplicate=True), Output('invert-cutoff-checkbox','value',allow_duplicate=True),
     Output('downsample-slider', 'disabled', allow_duplicate=True), Output('downsample-slider', 'value', allow_duplicate=True), # New
     Output('point-size-slider','disabled',allow_duplicate=True), Output('point-size-slider','value',allow_duplicate=True),
     Output('colormap-dropdown','disabled',allow_duplicate=True), Output('colormap-dropdown','value',allow_duplicate=True),
     Output('btn-save-processed','disabled',allow_duplicate=True), Output('btn-generate-stl','disabled',allow_duplicate=True),
     Output('btn-threejs-rotate','disabled',allow_duplicate=True), Output('btn-threejs-reset-view','disabled',allow_duplicate=True),
     Output('spacing-x','value',allow_duplicate=True), Output('spacing-y','value',allow_duplicate=True),
     Output('spacing-z','value',allow_duplicate=True), Output('mat-var-name-input','value',allow_duplicate=True),
     Output('stl-status-message','children',allow_duplicate=True), Output('upload-data','contents',allow_duplicate=True),
     Output('threejs-stats-footer','children',allow_duplicate=True)],
    [Input('btn-reset-data', 'n_clicks')], prevent_initial_call=True
)
def reset_ui(n):
    log_action("CB reset_ui: START")
    return (True,True,True,True, "Upload file.","", True,DEFAULT_THRESHOLD, True,False, # invert
            True,DEFAULT_DOWNSAMPLE_FACTOR, # downsample
            True,DEFAULT_POINT_SIZE, True,THREEJS_COLORMAPS[0], True,True,True,True,
            DEFAULT_SPACING,DEFAULT_SPACING,DEFAULT_SPACING, "","",None, "Load data.")

# --- Run ---
if __name__ == '__main__':
    log_action(f"Starting {APP_TITLE}")
    app.run(debug=False, host='0.0.0.0', port=8050)