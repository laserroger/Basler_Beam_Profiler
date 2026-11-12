"""Multiple individual D4sigma traces shared by desktop, HTTP and demo."""
import cv2
import sys
import numpy as np

WINDOW = 'Selected spot traces - 4 sigma diameter'


def bgr(hex_color):
    return tuple(int(hex_color[i:i+2],16) for i in (5,3,1))


def parse_y_limits(low, high):
    low, high = float(low), float(high)
    if not np.isfinite([low, high]).all() or not 0 <= low < high:
        raise ValueError('Use finite values with 0 <= Y min < Y max')
    return low, high


def window_render_geometry(title):
    """Read logical client bounds, independent of the image passed to imshow.

    Cocoa getWindowImageRect can follow the source image's pixel dimensions;
    multiplying those by backingScaleFactor each frame causes runaway growth.
    """
    if sys.platform == 'darwin':
        from AppKit import NSApplication
        for window in NSApplication.sharedApplication().windows():
            if str(window.title()) == title:
                bounds = window.contentView().bounds()
                return (max(1, round(bounds.size.width)),
                        max(1, round(bounds.size.height)),
                        max(1., float(window.backingScaleFactor())))
        return 1000, 720, 1.
    _, _, width, height = cv2.getWindowImageRect(title)
    return max(1, width), max(1, height), 1.


def render_sizes(data, y_limits=None, width=1000, height=720, pixel_ratio=1.):
    width, height = max(320, int(width)), max(280, int(height))
    pixel_ratio = max(1., min(4., float(pixel_ratio)))
    image = np.full((round(height*pixel_ratio), round(width*pixel_ratio), 3), (26, 22, 18), np.uint8)
    def device_point(p):
        return tuple(round(v*pixel_ratio) for v in p)
    def line(p1, p2, color, thickness=1):
        cv2.line(image, device_point(p1), device_point(p2), color, max(1,round(thickness*pixel_ratio)), cv2.LINE_AA)
    def circle(p, radius, color, thickness):
        cv2.circle(image, device_point(p), max(1,round(radius*pixel_ratio)), color,
                   -1 if thickness < 0 else max(1,round(thickness*pixel_ratio)), cv2.LINE_AA)
    def text(s, x, y, color=(225,225,225), scale=.55):
        cv2.putText(image, s, device_point((x,y)), cv2.FONT_HERSHEY_SIMPLEX, scale*pixel_ratio,
                    color, max(1,round(pixel_ratio)), cv2.LINE_AA)
    text('INDIVIDUAL SPOT TRACES | 4 sigma diameter', 20, 34, scale=min(.75,(width-40)/650))
    text('Major and minor diameters | matching spot colors',20,56,scale=min(.45,(width-40)/570))
    if data is None:
        text('Waiting for camera frame', 30, 95)
        return image
    traces=data.get('traces',[])
    if not traces:
        text('Click camera spots to add traces.', 20, 110, scale=.5)
        text('Click again to remove a trace.', 20, 145, scale=.5)
        return image
    columns=max(1,min(3,(width-40)//240))
    max_rows=max(1,min(4,(height-300)//22))
    legend_count=min(len(traces),columns*max_rows)
    rows=(legend_count+columns-1)//columns
    for j,t in enumerate(traces[:legend_count]):
        label=f"#{t['id']} ({t['x_sensor_px']:.0f},{t['y_sensor_px']:.0f})"
        if not t['valid']:
            label+=' - gap'
        text(label,20+(j%columns)*(width-40)/columns,80+(j//columns)*22,bgr(t['color']),scale=.5)
    if len(traces)>legend_count:
        text(f"+{len(traces)-legend_count} more selected",max(20,width-200),52,scale=.4)
    first_top=110+(rows-1)*22
    panel_height=max(25,(height-45-first_top-70)//2)
    first_bottom=first_top+panel_height
    second_top=first_bottom+70
    second_bottom=height-45
    def panel(title, key, top, bottom):
        left,right=85,width-25
        text(title,20,top-20,scale=min(.55,(width-40)/500))
        all_history=[h for t in traces for h in t['history']]
        ys=[h[k] for h in all_history for k in ('major_um','minor_um') if h[k] is not None]
        ymin,ymax=parse_y_limits(*y_limits) if y_limits is not None else (0.,max(ys,default=60)*1.15)
        origin=data.get('time_origin')
        if origin is None:
            origin=data['timestamp']
        xs=[h['timestamp']-origin for h in all_history]
        xs=[x for x in xs if x is not None]
        xmax=max(data["timestamp"]-origin,.01)
        xmin=max(0.,xmax-data.get("window_seconds",60.))
        if xmin==xmax:
            xmin,xmax=xmin-.5,xmax+.5
        X=lambda x:int(left+(x-xmin)/(xmax-xmin)*(right-left))
        Y=lambda y:int(np.clip(bottom-(y-ymin)/(ymax-ymin)*(bottom-top),-1000000,1000000))
        for y in np.linspace(ymin,ymax,3 if bottom-top<100 else 5):
            line((left,Y(y)),(right,Y(y)),(65,57,50),1)
            text(f'{y:.1f}',30,Y(y)+5,scale=.45)
        text('um',65,top-6,scale=.4)
        for x in np.linspace(xmin,xmax,5):
            text(f'{x:.1f}',X(x)-15,bottom+23,scale=.45)
        for trace in traces:
            color=bgr(trace['color'])
            previous=None
            for h in trace['history']:
                x=h['timestamp']-origin
                y=h[key]
                if x is None or y is None:
                    previous=None
                    continue
                point=(X(x),Y(y))
                if previous is not None:
                    visible,p1,p2=cv2.clipLine((left,top,right-left+1,bottom-top+1),previous,point)
                    if visible:
                        line(p1,p2,color,2)
                if ymin <= y <= ymax and left <= point[0] <= right:
                    circle(point,2,color,-1)
                previous=point
        xlabel='Elapsed time since reset (s)'
        label_width=cv2.getTextSize(xlabel,cv2.FONT_HERSHEY_SIMPLEX,.5,1)[0][0]
        text(xlabel,max(10,(left+right-label_width)/2),bottom+38,scale=.5)
    panel('Major axis: 4 sigma diameter', 'major_um', first_top, first_bottom)
    panel('Minor axis: 4 sigma diameter', 'minor_um', second_top, second_bottom)
    return image


class SizeControls:
    """Native controls for the desktop trace chart."""
    def __init__(self, session):
        import tkinter as tk
        from tkinter import ttk
        from .settings import prepare_gui
        self.limits = None
        self.root = tk.Toplevel(prepare_gui())
        self.root.title('Spot trace controls')
        self.root.protocol('WM_DELETE_WINDOW', self.root.withdraw)
        frame = ttk.Frame(self.root, padding=12)
        frame.pack()
        self.low, self.high = tk.StringVar(value='45'), tk.StringVar(value='65')
        for column, (label, variable) in enumerate([('Y min (um)',self.low),('Y max (um)',self.high)]):
            ttk.Label(frame,text=label).grid(row=0,column=column)
            entry=ttk.Entry(frame,textvariable=variable,width=12)
            entry.grid(row=1,column=column,padx=5)
            entry.bind('<Return>',lambda event:self.apply())
        ttk.Button(frame,text='Apply Y range',command=self.apply).grid(row=1,column=2,padx=5)
        ttk.Button(frame,text='Auto Y',command=self.auto).grid(row=1,column=3)
        ttk.Label(frame,text='Time window (s, 1-600)').grid(row=2,column=0,columnspan=2,pady=8)
        self.window = tk.StringVar(value='60')
        ttk.Entry(frame,textvariable=self.window,width=12).grid(row=3,column=0)
        ttk.Button(frame,text='Apply window',command=lambda:self.apply_window(session)).grid(row=3,column=1)
        ttk.Button(frame,text='Reset time',command=session.reset_trace_time).grid(row=3,column=2)
        self.error=ttk.Label(frame,text='')
        self.error.grid(row=4,column=0,columnspan=4,pady=8)

    def apply(self):
        try:
            self.limits=parse_y_limits(self.low.get(),self.high.get())
            self.error.configure(text='')
        except ValueError as exc:
            self.error.configure(text=str(exc))

    def auto(self):
        self.limits=None
        self.error.configure(text='')

    def apply_window(self,session):
        try:
            seconds=float(self.window.get())
            if not np.isfinite(seconds) or not 1<=seconds<=600:
                raise ValueError('Time window must be between 1 and 600 seconds')
            session.set_trace_window(seconds)
            self.error.configure(text='')
        except ValueError as exc:
            self.error.configure(text=str(exc))

    def destroy(self):
        self.root.destroy()
