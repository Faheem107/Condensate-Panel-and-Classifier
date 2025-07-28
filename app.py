from pathlib import Path
import pickle as pkl

import hvplot.pandas
import numpy as np
import pandas as pd
import panel as pn

from skimage.measure import profile_line

from bokeh.plotting import figure
from bokeh.models import LinearColorMapper

from puncta_tracking.io import read_movie
from puncta_tracking.track.xml import parse_trackmate_xml, extract_edges, extract_spots, extract_tracks, extract_tracks_graphs
from puncta_tracking.props import get_regionprops, match_spots_regionprops_dist
from puncta_tracking.phase_portrait.tracks import track_to_dataframe

def intensity_sum(region_mask,intensity_img):
    return np.sum(intensity_img[region_mask].flatten())


pkl_file = Path('/Volumes/Desk SSD/Lily/Actin Cortex Project/24 Hours Cap-1 RNAi/20240820/sum of normalized channels/bg subtract/results_usingsum/trackmate_tracks_dag.pkl')
movie_path = Path('/Volumes/Desk SSD/Lily/Actin Cortex Project/24 Hours Cap-1 RNAi/20240820/sum of normalized channels/Mergedwithbgsubtract.tif')
mask_path = Path('/Volumes/Desk SSD/Lily/Actin Cortex Project/24 Hours Cap-1 RNAi/20240820/sum of normalized channels/bg subtract/results_usingsum/ilastik_mask.tiff')

# Load movie and mask
movie, _ = read_movie(movie_path)
mask, _ = read_movie(mask_path)

# Load tracks
with open(pkl_file,'rb') as f:
    tracks = pkl.load(f)

# Remove merges or splits
tmp = {}
for track in tracks:
    if track.graph['NUMBER_MERGES'] + track.graph['NUMBER_SPLITS'] > 0:
        continue
    tmp[int(track.graph['TRACK_INDEX'])] = track_to_dataframe(track)[0]
tracks = tmp

# Channel plots
def get_track_movie(track,mov):
    min_r, min_c = int(track['bbox-0'].min())-20, int(track['bbox-1'].min()) -20
    max_r, max_c = int(track['bbox-2'].max()) +20, int(track['bbox-3'].max()) +20
    t = np.array(track['FRAME'],dtype=int)
    tup = sorted(t)
    new = tup
    for i in range(10):
        if new[-1] != 1500:
            new = new + [new[-1] + 1]
    return mov[tuple(sorted(new)),min_r:max_r+1,min_c:max_c+1]

# Kymograph
def get_kymo(mov):
    kymo = np.moveaxis(mov,0,-1)
    kymo = profile_line(kymo,(0,kymo.shape[1]//2),(kymo.shape[0]-1,kymo.shape[1]//2),linewidth=3)
    kymo = (kymo - kymo.min())/(kymo.max() - kymo.min())
    
    return kymo.T

# Channel and names
ch_names = ['Mask',*[f'Channel {ch+1}' for ch in range(movie.shape[1])]]

def _channel_plots(rel_time,track_id,ch):
    track = tracks[track_id]
    if ch == 0:
        # Mask
        img = get_track_movie(track,mask)
    else:
        # Movie
        img = get_track_movie(track,movie[:,ch-1])

    color = LinearColorMapper(palette='Greys256',low=img.min(),high=img.max())
    img = img[rel_time]

    w = 300
    h = int((img.shape[0]/img.shape[1])*w)
    
    fig = figure(height=h,width=w,x_range=(0,img.shape[1]),y_range=(img.shape[0],0))
    im = fig.image(image=[img],x=0,y=0,dw=img.shape[1],dh=img.shape[0],color_mapper=color,level="image")
    cb = im.construct_color_bar()
    fig.add_layout(cb,"below")
    fig.axis.visible = False
    
    return fig

def channel_plots(rel_time,track_id):
    return pn.Row(
        *[
            pn.Column(
                pn.pane.Str(ch_name,styles={'font-size':'15pt'},align='center'),
                pn.pane.Bokeh(_channel_plots(rel_time,track_id,ch),align='center',sizing_mode='scale_both')
            )
            for ch,ch_name in enumerate(ch_names)
        ]
    )   

def intensity_plot(track_id):
    df = tracks[track_id].sort_values('FRAME')
    intensity_cols = [f'intensity_sum-{ch}' for ch in range(movie.shape[1])]
    df = df[['FRAME',*intensity_cols,'TOTAL_INTENSITY_CH1']]
    new_cols = [f'Channel {ch+1}' for ch in range(movie.shape[1])]
    for col, new_col in zip(intensity_cols,new_cols):
        df[new_col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())
    df['Mask'] = df['TOTAL_INTENSITY_CH1']
    df['Mask'] = (df['Mask'] - df['Mask'].min())/(df['Mask'].max() - df['Mask'].min())
    df_hv = df.hvplot(
        x='FRAME',
        y=[*new_cols,'Mask'],
        xlabel='Time (frames)',
        ylabel='Normalized Intensity (au)',
        ylim=[0,1],
        xlim=[df['FRAME'].min(),df['FRAME'].max()],
    )
    df_hv.opts(legend_position='top_left')
    return pn.pane.HoloViews(df_hv,sizing_mode='stretch_width')

def _kymo_plots(track_id, ch):
    track = tracks[track_id]
    if ch == 0:
        # Mask
        img = get_track_movie(track,mask)
    else:
        # Movie
        img = get_track_movie(track,movie[:,ch-1])
    img = get_kymo(img)

    w = 300
    h = int((img.shape[0]/img.shape[1])*w)
    
    fig = figure(height=h,width=w,x_range=(0,img.shape[1]),y_range=(img.shape[0],0))
    fig.image(image=[img],x=0,y=0,dw=img.shape[1],dh=img.shape[0],palette="Greys256",level="image")
    fig.axis.visible = False

    return fig

def kymo_plots(track_id):
    return pn.Row(
        *[
            pn.Column(
                pn.pane.Str(ch_name,styles={'font-size':'15pt'},align='center'),
                pn.pane.Bokeh(_kymo_plots(track_id,ch),align='center',sizing_mode='scale_both')
            )
            for ch,ch_name in enumerate(ch_names)
        ]
    )

def intensity_kymo_plot(track_id):
    track = tracks[track_id]
    df = {'t':sorted(track['FRAME'])}
    cols = [f'Channel {ch+1}' for ch in range(movie.shape[1])]
    for ch in range(movie.shape[1]):
        kymo = get_kymo(get_track_movie(track,movie[:,ch]))
        df[f'Channel {ch+1}'] = kymo.mean(axis=1)
    df = pd.DataFrame(df)
    df_hv = df.hvplot(
        x='t',
        y=cols,
        xlabel='Time (frames)',
        ylabel='Average Intensity (au)',
        ylim=[0,1],
        xlim=[df['t'].min(),df['t'].max()],
    )
    df_hv.opts(legend_position='top_left')
    return pn.pane.HoloViews(df_hv,sizing_mode='stretch_width')

def intensity_bound_plot(track_id):
    track = tracks[track_id]
    df = {'t':sorted(track['FRAME'])}
    cols = [f'Channel {ch+1}' for ch in range(movie.shape[1])]
    for ch in range(movie.shape[1]):
        col = f'Channel {ch+1}'
        img = get_track_movie(track,movie[:,ch])
        df[col] = img.sum(axis=(-2,-1))
        df[col] = (df[col] - df[col].min())/(df[col].max() - df[col].min())
    df = pd.DataFrame(df)
    df_hv = df.hvplot(
        x='t',
        y=cols,
        xlabel='Time (frames)',
        ylabel='Normalized Intensity (au)',
        ylim=[0,1],
        xlim=[df['t'].min(),df['t'].max()],
    )
    df_hv.opts(legend_position='top_left')
    return pn.pane.HoloViews(df_hv,sizing_mode='stretch_width')

pn.extension()

# Track number
track_id_opts = sorted(tracks.keys(),key=lambda t: len(tracks[t]),reverse=True)
track_id = pn.widgets.Select(
    description='Select Track ID',
    name='Track ID',
    options=track_id_opts,
)

# Time slider
rel_time = pn.widgets.IntSlider(
    name='Relative Time',
    start= -10,
    end=len(tracks[track_id_opts[0]])-1 +10,
    step=1,
    value=0,
)

# Set time slider according to track
def set_max_rel_time(track_id):
    rel_time.value = 0
    max_rel_time = len(tracks[track_id]) - 1
    rel_time.end = max_rel_time + 10

pn.bind(set_max_rel_time,track_id,watch=True)

# Bind figures
ch_plots = pn.bind(channel_plots,rel_time=rel_time,track_id=track_id)
int_plot = pn.bind(intensity_plot,track_id=track_id)
ky_plots = pn.bind(kymo_plots,track_id=track_id)
#int_ky_plot = pn.bind(intensity_kymo_plot,track_id=track_id)
int_bbox_plot = pn.bind(intensity_bound_plot,track_id=track_id)

# Lay out widgets and panels
# Widgets
widget_row = pn.Row(
    rel_time,
    pn.Spacer(sizing_mode='stretch_width'),
    track_id
).servable()

# Images
pn.Column(
    pn.pane.Markdown("# Over Time",align='center'),
    pn.Row(ch_plots,int_plot),
    pn.pane.Markdown("# Kymograph",align='center'),
    pn.Row(ky_plots),
).servable()
