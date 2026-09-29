"""Optional native plot with typed, fixed axis limits."""
import math
import time
import tkinter as tk
from tkinter import ttk
import cv2
from .settings import prepare_gui
import numpy as np


def parse_limits(lower, upper):
    low, high = float(lower), float(upper)
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError('Enter finite values with Y min < Y max')
    return low, high


class SpacingWindow:
    def __init__(self, on_close):
        self.limits = (1., 1.4)
        self.last_draw = 0.
        self.root = tk.Toplevel(prepare_gui())
        self.root.title('Spot spacing')
        self.root.geometry('+800+80')
        self.root.protocol('WM_DELETE_WINDOW', on_close)
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill='x')
        self.low, self.high = tk.StringVar(value='1.0'), tk.StringVar(value='1.4')
        for label, var in [('Y min (mrad)', self.low), ('Y max (mrad)', self.high)]:
            ttk.Label(controls, text=label).pack(side='left', padx=4)
            entry = ttk.Entry(controls, textvariable=var, width=9)
            entry.pack(side='left')
            entry.bind('<Return>', self.apply)
        ttk.Button(controls, text='Apply', command=self.apply).pack(side='left', padx=8)
        self.error = ttk.Label(self.root, text='')
        self.error.pack()
        self.canvas = tk.Canvas(self.root, width=760, height=600, background='white', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True)
        self.result = None
        self.canvas.bind('<Configure>', lambda event: self.draw() if self.result else None)

    def apply(self, event=None):
        try:
            self.limits = parse_limits(self.low.get(), self.high.get())
            self.error.configure(text='')
            self.last_draw = 0.
            self.draw()
        except ValueError:
            self.error.configure(text='Enter numbers with Y min < Y max')

    def update(self, result):
        now = time.monotonic()
        if now-self.last_draw < 1/30:
            return
        self.last_draw = now
        self.result = result
        self.draw()

    def draw(self):
        c = self.canvas
        c.delete('all')
        d = self.result
        if d is None:
            return
        W,H = max(c.winfo_width(),300), max(c.winfo_height(),250)
        c.create_text(W/2,24,text='Absolute angular spacing |Δθ|',font=('Helvetica',17))
        c.create_text(W/2,51,text='Pair 1 starts at the rightmost spot; count toward the left',font=('Helvetica',11))
        if not d.get('valid'):
            c.create_text(W/2,H/2,text=d.get('error','Waiting'),width=W-50)
            return
        L,R,T,B = 85,W-30,95,H-100
        x=np.asarray(d['pair_index']); y=np.asarray(d['response'])
        xmin,xmax = (x[0],x[-1]) if len(x)>1 else (0,2)
        low,high = self.limits
        X=lambda v:L+(v-xmin)/(xmax-xmin)*(R-L)
        Y=lambda v:B-(v-low)/(high-low)*(B-T)
        for v in np.linspace(low,high,6):
            py=Y(v)
            c.create_line(L,py,R,py,fill='#dddddd')
            c.create_text(L-12,py,text=f'{v:.3f}',anchor='e',font=('Helvetica',12))
        for v in np.unique(np.linspace(x[0],x[-1],min(len(x),8)).round().astype(int)):
            c.create_text(X(v),B+22,text=str(v),font=('Helvetica',12))
        c.create_text(W/2,B+50,text='Adjacent-pair index',font=('Helvetica',13))
        c.create_text(22,(T+B)/2,text='|Δθ| (mrad)',angle=90,font=('Helvetica',13))
        trend=d['trend']
        if len(x)>1:
            pt1=(int(X(x[0])),int(np.clip(Y(trend[0]),-1000000,1000000)))
            pt2=(int(X(x[-1])),int(np.clip(Y(trend[-1]),-1000000,1000000)))
            visible,p1,p2=cv2.clipLine((L,T,R-L+1,B-T+1),pt1,pt2)
            if visible:c.create_line(*p1,*p2,fill='#c02c76',width=2)
        for v,z in zip(x,y):
            if low<=z<=high:
                px,py=X(v),Y(z)
                c.create_oval(px-3,py-3,px+3,py+3,fill='#147dc0',outline='')
        c.create_text(W/2,H-20,text=f"{d['detected_spots']} spots · analysis {d['analysis_ms']:.2f} ms · single shot",font=('Helvetica',11))

    def destroy(self):
        self.root.destroy()
