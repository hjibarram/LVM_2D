#!/usr/bin/env python
import glob, os,sys,timeit
import matplotlib
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import warnings
import types
import astropy.cosmology.funcs as cd
import importlib.machinery
import os.path as ptt
#from progressbar import ProgressBar
import math
from scipy.ndimage.filters import gaussian_filter1d as filt1d
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.wcs import WCS
from scipy.interpolate.interpolate import interp1d
#from __builtin__ import True
warnings.filterwarnings("ignore")
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy import units as u
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs.utils import pixel_to_skycoord
from scipy.special import erf
from scipy.optimize import curve_fit
from matplotlib import colors
import emcee
from scipy.special import gamma, gammaincinv, gammainc
from numpy import random as ran

def cosmic_rays(arrayf,n_cr=100,d_cr=5):
    nf,ng=arrayf.shape
    array1=np.zeros([nf,ng])    
    nc=n_cr+np.int(ran.randn(1)[0]*d_cr) 
    xof=ran.rand(nc)*nf
    yof=ran.rand(nc)*ng
    thet=ran.rand(nc)*(90.0+90.0)-90.0
    phi=ran.rand(nc)*(90.0-5.0)+5.0
    deep=10.0
    lent=deep/np.sin(phi*np.pi/180.0)
    for k in range(0, nc):
        lx=np.int(lent[k]*np.cos(thet[k]*np.pi/180.0))+1        
        x_tc=np.arange(lx)+xof[k]
        cof=yof[k]-np.tan(thet[k]*np.pi/180.0)*xof[k]       
        y_tc=np.tan(thet[k]*np.pi/180.0)*x_tc+cof
        for i in range(0, lx):
            xt1=np.int(x_tc[i])-1
            xt2=np.int(x_tc[i])#+1
            yt1=np.int(y_tc[i])-1
            yt2=np.int(y_tc[i])#+1
            if yt1 < 0:
                yt1=0
            if yt2 < 0:
                yt2 =0
            if yt1 > ng:
                yt1=ng
            if yt2 > ng:
                yt2=ng
            if xt1 < 0:
                xt1=0
            if xt2 < 0:
                xt2=0
            if xt1 > nf:
                xt1=nf
            if xt2 > nf:
                xt2=nf
            array1[xt1:xt2,yt1:yt2]=100000.0#60000.0
    dv=0.5
    PSF=Gaussian2DKernel(x_stddev=dv,y_stddev=dv)
    array1=convolve(array1, PSF)#, mode='full')#,  boundary='symm')#photo_a#
    arrayf=arrayf+array1
    arrayf[np.where(arrayf >= 60000.0)]=60000.0
    return arrayf

def id_str(id,n_z=2):
    id=int(np.float_(id))
    if n_z < 2 or n_z > 11:
        n_z=2
    if n_z == 2:
        if id < 10:
            idt='0'+str(id)
        else:
            idt=str(id)
    elif n_z == 3:
        if id < 10:
            idt='00'+str(id)
        elif id < 100:
            idt='0'+str(id)
        else:
            idt=str(id)
    elif n_z == 4:        
        if id < 10:
            idt='000'+str(id)
        elif id < 100:
            idt='00'+str(id)
        elif id < 1000:
            idt='0'+str(id)
        else:
            idt=str(id)
    elif n_z == 5:        
        if id < 10:
            idt='0000'+str(id)
        elif id < 100:
            idt='000'+str(id)
        elif id < 1000:
            idt='00'+str(id)
        elif id < 10000:
            idt='0'+str(id)
        else:
            idt=str(id)
    elif n_z == 6:        
        if id < 10:
            idt='00000'+str(id)
        elif id < 100:
            idt='0000'+str(id)
        elif id < 1000:
            idt='000'+str(id)
        elif id < 10000:
            idt='00'+str(id)
        elif id < 100000:
            idt='0'+str(id)
        else:
            idt=str(id)
    elif n_z == 7:        
        if id < 10:
            idt='000000'+str(id)
        elif id < 100:
            idt='00000'+str(id)
        elif id < 1000:
            idt='0000'+str(id)
        elif id < 10000:
            idt='000'+str(id)
        elif id < 100000:
            idt='00'+str(id)
        elif id < 1000000:
            idt='0'+str(id)
        else:
            idt=str(id)
    elif n_z == 8:        
        if id < 10:
            idt='0000000'+str(id)
        elif id < 100:
            idt='000000'+str(id)
        elif id < 1000:
            idt='00000'+str(id)
        elif id < 10000:
            idt='0000'+str(id)
        elif id < 100000:
            idt='000'+str(id)
        elif id < 1000000:
            idt='00'+str(id)
        elif id < 10000000:
            idt='0'+str(id)
        else:
            idt=str(id)
    elif n_z == 9:        
        if id < 10:
            idt='00000000'+str(id)
        elif id < 100:
            idt='0000000'+str(id)
        elif id < 1000:
            idt='000000'+str(id)
        elif id < 10000:
            idt='00000'+str(id)
        elif id < 100000:
            idt='0000'+str(id)
        elif id < 1000000:
            idt='000'+str(id)
        elif id < 10000000:
            idt='00'+str(id)
        elif id < 100000000:
            idt='0'+str(id)
        else:
            idt=str(id)
    elif n_z == 10:        
        if id < 10:
            idt='000000000'+str(id)
        elif id < 100:
            idt='00000000'+str(id)
        elif id < 1000:
            idt='0000000'+str(id)
        elif id < 10000:
            idt='000000'+str(id)
        elif id < 100000:
            idt='00000'+str(id)
        elif id < 1000000:
            idt='0000'+str(id)
        elif id < 10000000:
            idt='000'+str(id)
        elif id < 100000000:
            idt='00'+str(id)
        elif id < 1000000000:
            idt='0'+str(id)
        else:
            idt=str(id)
    elif n_z == 11:        
        if id < 10:
            idt='0000000000'+str(id)
        elif id < 100:
            idt='000000000'+str(id)
        elif id < 1000:
            idt='00000000'+str(id)
        elif id < 10000:
            idt='0000000'+str(id)
        elif id < 100000:
            idt='000000'+str(id)
        elif id < 1000000:
            idt='00000'+str(id)
        elif id < 10000000:
            idt='0000'+str(id)
        elif id < 100000000:
            idt='000'+str(id)
        elif id < 1000000000:
            idt='00'+str(id)
        elif id < 10000000000:
            idt='0'+str(id)
        else:
            idt=str(id)
    return idt


def read_op_fib(cart,cam,dir='libs/'):
    if cart < 0 and cart > 18:
        cart=16
    #if not 'b1' in cam:
    #    if not 'b2' in cam:
    #        if not 'r1' in cam:
    #            if not 'r2' in cam:
    #                if not 'z1' in cam:
    #                    if not 'z2' in cam:
    #                        cam='b1' 
    file=dir+'opFibers.par'  
    f=open(file,'r')
    for line in f:
        if 'FIBERPARAM' in line:
            dat=line.replace('\n','').split('{')
            data1=dat[0].split(' ')
            data1=list(filter(None,data1))
            if len(data1) > 2:
                car=np.int(data1[1])
                lap=data1[2]
                #print(car,lap,cart,cam)
                if car == cart and cam == lap:
                    print(car,lap)
                    data2=dat[1].replace('}','').split(' ')
                    data2=filter(None,data2)
                    space=np.array([np.float(val) for val in data2])
                    data3=dat[2].replace('}','').split(' ')
                    data3=filter(None,data3)
                    bspa=np.array([np.float(val) for val in data3])
                    bspa[0]=bspa[0]#-279.0
    space=space
    bspa=bspa
    #print(space)
    return space,bspa

def wfits_ext(name,hlist):
    sycall("rm "+name+'.gz')
    if ptt.exists(name) == False:
        hlist.writeto(name)
    else:
        name1=name.replace("\ "," ")
        name1=name1.replace(" ","\ ")
        sycall("rm "+name1)
        hlist.writeto(name)

def sycall(comand):
    import os
    linp=comand
    os.system(comand)

def get_focus(dir1='',name='focus_lvm',dsx=0.5,dsy=0.5,rho=0.05,vt1=0.3,vt2=0.3,vt3=0.1,lt=1500.):
    nf=4080#4224
    ng=4120#4352
    arrayf=np.zeros([3,nf,ng])
    xo=nf/2
    yo=ng/2
    for i in range(0, nf):
        for j in range(0, ng):
            rad=np.sqrt((i-xo)**2.+(j-yo)**2.)
            val1=vt1*(rad/lt)**2+dsx
            val2=vt2*(rad/lt)**2+dsy
            val3=vt3*(rad/lt)**2+rho
            arrayf[0,i,j]=val1
            arrayf[1,i,j]=val2
            arrayf[2,i,j]=val3    
    h1=fits.PrimaryHDU(arrayf)
    hlist=fits.HDUList([h1])
    hlist.update_extend()
    out_fit=dir1+name+'.fits'
    wfits_ext(out_fit,hlist)
    sycall('gzip -f '+out_fit)    
    
def raw_exp_bhm(spec,fibf,base_name,fc=[1.0,1.0],n_cr=130,d_cr=5,type="blue",cam=1,dir1='./',nfib=500,mjd='00000',plate='0000',exp=0,flb='s',expt=900.0,ra0=0.0,dec0=0.0,expof=0.0):       
    nx,ny=spec.shape
    cam=str(int(cam))
    nf=4080#4224
    ng=4120#4352
    arrayf_1=np.zeros([nf,ng])
    b_arrayf_1=np.zeros([nf,ng])
    if type == "blue":
        p_arrayf_1=np.zeros([4012,4096])#4112,4096
        f_arrayf_1=np.ones([4012,4096])
        b1_arrayf_1=np.zeros([4012,4096])
    if type == "red":
        p_arrayf_1=np.zeros([4128,4114])    
        f_arrayf_1=np.ones([4128,4114])
        b1_arrayf_1=np.zeros([4128,4114])
    if type == "nir":
        p_arrayf_1=np.zeros([4128,4114])    
        f_arrayf_1=np.ones([4128,4114])
        b1_arrayf_1=np.zeros([4128,4114])    
    nt=np.argsort(fibf)
    if type == "blue":
        let=800#800
        let2=300
        ty='b'
        fibs1,bunds1=read_op_fib(1,'b'+cam)
        try:
            focus=fits.getdata('libs/focus_lvm_blue'+cam+'.fits.gz', 0, header=False)
            print('Using focus file')
        except:
            focus=np.ones([3,nf,ng])
            focus[0,:,:]=1.0
            focus[1,:,:]=0.9
            focus[2,:,:]=0.0
    if type == "red":
        let=800#620
        let2=300
        ty='r'
        fibs1,bunds1=read_op_fib(1,'r'+cam)
        try:
            focus=fits.getdata('libs/focus_lvm_red'+cam+'.fits.gz', 0, header=False)
            print('Using focus file')
        except:
            focus=np.ones([3,nf,ng])
            focus[0,:,:]=1.0
            focus[1,:,:]=0.9
            focus[2,:,:]=0.0
    if type == "nir":
        let=800#620
        let2=300
        ty='z'
        fibs1,bunds1=read_op_fib(1,'z'+cam)    
        try:
            focus=fits.getdata('libs/focus_lvm_nir'+cam+'.fits.gz', 0, header=False)
            print('Using focus file')
        except:
            focus=np.ones([3,nf,ng])
            focus[0,:,:]=1.0
            focus[1,:,:]=0.9
            focus[2,:,:]=0.0
            
    for i in range(0, nfib):#len(fibf)):
        if '1' in cam:
            init_v=0
        elif '2' in cam:
            init_v=nfib
        elif '3' in cam:
            init_v=nfib*2
        else:
            init_v=0
        it=i
        iy=i+init_v
        r=let*(2.7+0.0)
        then=np.arcsin((it-(nfib/2.))/r)                 
        dr=np.cos(then)*r-let*(1.7+0.0)
        dx=np.int(np.round(dr))
        #print(i,iy,nt[iy],fibf[nt[iy]])
        spect=spec[:,nt[iy]]
        npix1=100
        npix2=100#200
        y_e=spect[0]/np.float(npix1)*np.arange(npix1)
        y_r=spect[::-1][0]/np.float(npix2)*np.arange(npix2)
        spect=np.concatenate((y_e,spect))
        spect=np.concatenate((spect,y_r[::-1]))
        dx=dx-npix1-150        
        nx=len(spect)
        dsi=np.ones(nx)#*0.5
        indt=it % 36
        itt=np.int(it/36)
        if indt == 0:
            if it > 0:
                #dt=16.4541#+np.abs(ran.randn(1)*0.7)-0.1
                dt=bunds1[itt]+fibs1[itt]#
            else:
                if type == 'blue':
                    dt=36.0+bunds1[0]#+36.6-2.0#+6.578*4.5#+280#bunds1[0]+
                if type == 'red':
                    dt=36.0+bunds1[0]#+36.6-18.3+9.15
                if type == 'nir':
                    dt=36.0+bunds1[0]#+36.6-18.3+9.15
                dg=0.0
        else:
            dt=6.578
            #print(fibs1)
            dt=fibs1[itt]
        dg=dg+dt

        dy=np.int(np.round(dg))
        off=dy-dg
        dc=10.0
        #nxt=np.int(dc*2+1)
        #x_t=np.arange(nxt)*1.0-dc+off
        #x_t=np.array([x_t]*nx)
        #ds_t=np.array([dsi]*nxt).T
        #At=np.array([spect]*nxt).T
        #spec_t=np.exp(-0.5*(x_t/ds_t)**2.0)/(np.sqrt(2.0*np.pi)*ds_t)*At
        
        dtt=0
        tta=0
        if tta == 0:
            nxt=np.int(dc*2+1)
            nyt=np.int(dc*2+1)
            x_t=np.arange(nxt)*1.0-dc+off
            y_t=np.arange(nyt)*1.0-dc
            x_t=np.array([x_t]*nyt)
            At=np.array([spect]*nxt).T
            nsx,nsy=At.shape
            spec_tt=np.zeros([int(nsx+2*dc),nsy])
            #print(i,spec_tt.shape)
            for j in range(0, nx):
                xo=dx+j+int(dc)
                yo=dy+int(dc)
                ds_x=np.array([np.ones(nyt)*focus[0,xo,yo]]*nxt).T
                ds_y=np.array([np.ones(nyt)*focus[1,xo,yo]]*nxt).T
                rho=np.array([np.ones(nyt)*focus[2,xo,yo]]*nxt).T
                Att=np.array([np.ones(nyt)*spect[j]]*nxt).T
                spec_ttt=np.exp(-0.5/(1-rho**2)*((x_t/ds_x)**2.0+(y_t/ds_y)**2.0-2*rho*(y_t/ds_y)*(x_t/ds_x)))/(2*np.pi*ds_x*ds_y*np.sqrt(1-rho**2))*Att
                spec_tt[j:j+int(2*dc+1),0:int(2*dc+1)]=spec_ttt+spec_tt[j:j+int(2*dc+1),0:int(2*dc+1)]
            spec_t=spec_tt    
            dtt=int(dc)
        
        y1=0
        y2=nxt
        x1=0
        x2=nx
        #print(i)
        #print(nxt)
        #print(spec_t.shape,dy+y2)
        #print(dx,nx)
        #print(dx+x1-dtt,dx+x2+dtt,dy+y1,dy+y2)
        #print(spec_t.shape,arrayf_1[dx+x1-dtt:dx+x2+dtt,dy+y1:dy+y2].shape)
        arrayf_1[dx+x1-dtt:dx+x2+dtt,dy+y1:dy+y2]=spec_t+arrayf_1[dx+x1-dtt:dx+x2+dtt,dy+y1:dy+y2]    
    
    if type == "blue":
        bias_1=2171.0
        sig_1=2.0*fc[0]*0.56
        gain_1=[1.048, 1.048, 1.018, 1.006]
        arrayf_1[0:2039,0:2059]=arrayf_1[0:2039,0:2059]/gain_1[0]+ran.randn(2039,2059)*sig_1+bias_1+35.0#2206  0:2111,0:2175
        arrayf_1[2039:4080,0:2059]=arrayf_1[2039:4080,0:2059]/gain_1[1]+ran.randn(2041,2059)*sig_1+bias_1+0.0#2171       
        arrayf_1[0:2039,2059:4120]=arrayf_1[0:2039,2059:4120]/gain_1[2]+ran.randn(2039,2061)*sig_1+bias_1-30.0#2141
        arrayf_1[2039:4080,2059:4120]=arrayf_1[2039:4080,2059:4120]/gain_1[3]+ran.randn(2041,2061)*sig_1+bias_1+129.0#2300
        b_arrayf_1[0:2039,0:2059]=ran.randn(2039,2059)*sig_1+bias_1+35.0#2206
        b_arrayf_1[2039:4080,0:2059]=ran.randn(2041,2059)*sig_1+bias_1+0.0#2171       
        b_arrayf_1[0:2039,2059:4120]=ran.randn(2039,2061)*sig_1+bias_1-30.0#2141
        b_arrayf_1[2039:4080,2059:4120]=ran.randn(2041,2061)*sig_1+bias_1+129.0#2300
        b1_arrayf_1=ran.randn(4112,4096)*sig_1+2.0
    if type == "red":
        bias_1=2120.0#2490.0
        sig_1=2.0*fc[0]*0.56
        gain_1=[1.9253, 1.5122, 1.4738, 1.5053]
        arrayf_1[0:2039,0:2059]=arrayf_1[0:2039,0:2059]/gain_1[0]+ran.randn(2039,2059)*sig_1+bias_1+5.0#+2495-4.0
        arrayf_1[2039:4080,0:2059]=arrayf_1[2039:4080,0:2059]/gain_1[1]+ran.randn(2041,2059)*sig_1+bias_1-4.0#2490-4.0
        arrayf_1[0:2039,2059:4120]=arrayf_1[0:2039,2059:4120]/gain_1[2]+ran.randn(2039,2061)*sig_1+bias_1+55.0#+2545-4.0
        arrayf_1[2039:4080,2059:4120]=arrayf_1[2039:4080,2059:4120]/gain_1[3]+ran.randn(2041,2061)*sig_1+bias_1+100.0#246.0#2740-4.0
        b_arrayf_1[0:2039,0:2059]=ran.randn(2039,2059)*sig_1+bias_1+5.0#+2495-4.0
        b_arrayf_1[2039:4080,0:2059]=ran.randn(2041,2059)*sig_1+bias_1-4.0#2490-4.0
        b_arrayf_1[0:2039,2059:4120]=ran.randn(2039,2061)*sig_1+bias_1+55.0#+2545-4.0
        b_arrayf_1[2039:4080,2059:4120]=ran.randn(2041,2061)*sig_1+bias_1+100.0#246.0#2740-4.0
        b1_arrayf_1=ran.randn(4128,4114)*sig_1+2.0
    if type == "nir":
        bias_1=2120.0#2490.0
        sig_1=2.0*fc[0]*0.56
        gain_1=[1.9253, 1.5122, 1.4738, 1.5053]
        arrayf_1[0:2039,0:2059]=arrayf_1[0:2039,0:2059]/gain_1[0]+ran.randn(2039,2059)*sig_1+bias_1+5.0#+2495-4.0
        arrayf_1[2039:4080,0:2059]=arrayf_1[2039:4080,0:2059]/gain_1[1]+ran.randn(2041,2059)*sig_1+bias_1-4.0#2490-4.0
        arrayf_1[0:2039,2059:4120]=arrayf_1[0:2039,2059:4120]/gain_1[2]+ran.randn(2039,2061)*sig_1+bias_1+55.0#+2545-4.0
        arrayf_1[2039:4080,2059:4120]=arrayf_1[2039:4080,2059:4120]/gain_1[3]+ran.randn(2041,2061)*sig_1+bias_1+100.0#246.0#2740-4.0
        b_arrayf_1[0:2039,0:2059]=ran.randn(2039,2059)*sig_1+bias_1+5.0#+2495-4.0
        b_arrayf_1[2039:4080,0:2059]=ran.randn(2041,2059)*sig_1+bias_1-4.0#2490-4.0
        b_arrayf_1[0:2039,2059:4120]=ran.randn(2039,2061)*sig_1+bias_1+55.0#+2545-4.0
        b_arrayf_1[2039:4080,2059:4120]=ran.randn(2041,2061)*sig_1+bias_1+100.0#246.0#2740-4.0
        b1_arrayf_1=ran.randn(4128,4114)*sig_1+2.0    
    
    sycall('mkdir -p '+dir1+'lvm')        
    dir0='lvm/'+str(mjd)#+'/'+plate.split('-')[0]#+'-'+mjd        
    sycall('mkdir -p '+dir1+'lvm/'+str(mjd))
    #sycall('mkdir -p '+dir1+dir0)        
    dir0f=dir0+'/raw_mock'
    sycall('mkdir -p '+dir1+dir0f)  
    dir0f=dir0f+'/'
    arrayf_1=cosmic_rays(arrayf_1,n_cr=n_cr,d_cr=d_cr)-32768.0
    arrayf_1[np.where(arrayf_1 > 32767)]=32767.0
    arrayf_1=np.array(arrayf_1,dtype='int16')
    h1=fits.PrimaryHDU(arrayf_1)
    h=h1.header
    h["NAXIS"]=2 
    h["NAXIS1"]=ng
    h["NAXIS2"]=nf
    h=row_data_header_bhm(h,plate,mjd,exp,ty+'',flb=flb,expt=expt,ra=ra0,dec=dec0,expof=expof)
    hlist=fits.HDUList([h1])
    hlist.update_extend()
    out_fit=dir1+dir0f+base_name+'-'+ty+cam+'-'+id_str(exp,n_z=8)+'.fit'
    wfits_ext(out_fit,hlist)
    sycall('gzip -f '+out_fit)


    if flb == 'f':
        sycall('mkdir -p '+dir1+'lvm/flats')
        sycall('mkdir -p '+dir1+'lvm/biases')
        h1=pyf.PrimaryHDU(b_arrayf_1)
        h=h1.header
        h["NAXIS"]=2 
        h["NAXIS1"]=ng
        h["NAXIS2"]=nf
        h=row_data_header2(h,mjd)
        hlist=pyf.HDUList([h1])
        hlist.update_extend()
        out_fit=dir1+'lvm/biases/boss_pixbias-'+str(mjd)+'-'+ty+cam+'.fits' 
        wfits_ext(out_fit,hlist)
        sycall('gzip -f '+out_fit)
    

        h1=pyf.PrimaryHDU(f_arrayf_1)
        h=h1.header
        hlist=pyf.HDUList([h1])
        hlist.update_extend()
        out_fit=dir1+'lvm/flats/pixflatave-'+str(mjd)+'-'+ty+cam+'.fits' 
        wfits_ext(out_fit,hlist)
        sycall('gzip -f '+out_fit) 
        
        h1=pyf.PrimaryHDU(p_arrayf_1)
        h=h1.header
        hlist=pyf.HDUList([h1])
        hlist.update_extend()
        out_fit=dir1+'lvm/flats/badpixels-'+str(mjd)+'-'+ty+cam+'.fits' 
        wfits_ext(out_fit,hlist)
        sycall('gzip -f '+out_fit)
        
        h1=pyf.PrimaryHDU(b1_arrayf_1)
        h=h1.header
        hlist=pyf.HDUList([h1])
        hlist.update_extend()
        out_fit=dir1+'lvm/biases/pixbiasave-00001'+'-'+ty+cam+'.fits'
        wfits_ext(out_fit,hlist)


def row_data_header_bhm(h,plate,mjd,exp,typ,flb='s',ra=0.0,dec=0.0,azim=180.0,alt=90.0,expt=900.0,expof=0.0):  
    if flb == 's':
        flab='science '
    elif flb == 'a':
        flab='arc     '
        expt=4.0
    elif flb == 'f':
        flab='flat    '  
        expt=25.0                                        
    h["TELESCOP"]= 'SDSS 2-5m'                                                           
    h["FILENAME"]= 'sdR-'+typ+'-'+id_str(exp,n_z=8)+'.fit'                                                 
    h["CAMERAS"] = typ+'      '                                                            
    h["EXPOSURE"]=  exp                                                  
    h["V_BOSS"]  = ('v4_0    ' ,'Active version of the BOSS ICC')              
    h["CAMDAQ"]  = '1.5.0:37'                                                            
    h["SUBFRAME"]= ('' ,'the subframe readout command')                                    
    h["ERRCNT"]  = 'NONE    '                                                            
    h["SYNCERR"] = 'NONE    '                                                            
    h["SLINES"]  = 'NONE    '                                                            
    h["PIXERR"]  = 'NONE    '                                                            
    h["PLINES"]  = 'NONE    '                                                            
    h["PFERR"]   = 'NONE    '                                                            
    h["DIDFLUSH"]= (True ,'CCD was flushed before integration')          
    h["FLAVOR"]  = (flab,'exposure type, SDSS spectro style')            
    h["MJD"]     = (np.int(mjd) ,'APO fMJD day at start of exposure')          
    h["TAI-BEG"] = ((np.float(mjd)+0.25)*24.0*3600.0+expof,'MJD(TAI) seconds at start of integration')  
    h["DATE-OBS"]= ('2012-03-20T06:00:00','TAI date at start of integration')           
    h["V_GUIDER"]= ('v3_4    ','version of the current guiderActor')            
    h["V_SOP"]   = ('v3_8_1  ','version of the current sopActor')           
    h["NAME"]    = (plate+'-'+mjd+'-01','The name of the currently loaded plate')     
    h["CONFIID"] = (plate,'The currently FPS configuration')              
    h["CARTID"]  = (16,'The currently loaded cartridge')                 
    h["MAPID"]   = (1,'The mapping version of the loaded plate') 
    h["POINTING"]= ('A       ','The currently specified pointing')               
    h["CONFTYP"]= ('BOSS    ','Type of plate (e.g. BOSS, APOGEE, BA')
    h["SRVYMODE"]= ('None    ','Survey leading this observation and its mode')
    h["OBJSYS"]  = ('ICRS    ','The TCC objSys')             
    h["RA"]      = (ra,'RA of telescope boresight (deg)')             
    h["DEC"]     = (dec,'Dec of telescope boresight (deg)')              
    h["RADEG"]   = (ra+0.704,'RA of telescope pointing(deg)')                 
    h["DECDEG"]  = (dec+0.083,'Dec of telescope pointing (deg)')                
    h["SPA"]     = (-158.0698343797722,'TCC SpiderInstAng')                              
    h["ROTTYPE"] = ('Obj     ','Rotator request type')                           
    h["ROTPOS"]  = (0.0,'Rotator request position (deg)')            
    h["BOREOFFX"]= (0.0,'TCC Boresight offset, deg')               
    h["BOREOFFY"]= (0.0,'TCC Boresight offset, deg')                    
    h["ARCOFFX"] = (-8.8999999999999E-05,'TCC ObjArcOff, deg')                    
    h["ARCOFFY"] = (-0.000807,'TCC ObjArcOff, deg')                           
    h["CALOFFX"] = (0.0,'TCC CalibOff, deg')                           
    h["CALOFFY"] = (0.0,'TCC CalibOff, deg')                            
    h["CALOFFR"] = (0.0,'TCC CalibOff, deg')                            
    h["GUIDOFFX"]= (0.0,'TCC GuideOff, deg')                            
    h["GUIDOFFY"]= (0.0,'TCC GuideOff, deg')                            
    h["GUIDOFFR"]= (0.052684,'TCC GuideOff, deg')                            
    h["AZ"]      = (azim,'Azimuth axis pos. (approx, deg)')           
    h["ALT"]     = (alt,'Altitude axis pos. (approx, deg)')              
    h["IPA"]     = (21.60392,'Rotator axis pos. (approx, deg)')             
    h["FOCUS"]   = (10.7512,'User-specified focus offset (um)')              
    h["M2PISTON"]= (357.36,'TCC SecOrient')             
    h["M2XTILT"] = (7.19,'TCC SecOrient')                                
    h["M2YTILT"] = (-18.2,'TCC SecOrient')                                
    h["M2XTRAN"] = (24.89,'TCC SecOrient')                                
    h["M2YTRAN"] = (-110.34,'TCC SecOrient')                                
    h["M2ZROT"]  = (-19.77,'TCC SecOrient')                                
    h["M1PISTON"]= (-949.28,'TCC PrimOrient')                                
    h["M1XTILT"] = (-24.31,'TCC PrimOrient')                               
    h["M1YTILT"] = (6.14,'TCC PrimOrient')                               
    h["M1XTRAN"] = (356.01,'TCC PrimOrient')                               
    h["M1YTRAN"] = (1322.6,'TCC PrimOrient')                               
    h["M1ZROT"]  = (0.03,'TCC PrimOrient')                  
    h["SCALE"]   = (1.000096,'User-specified scale factor')              
    h["V_APO"]   = ('trunk+svn158476M','version of the current apoActor')              
    h["PRESSURE"]=21.413                                                  
    h["WINDD"]   =286.0                                                  
    h["WINDS"]   =18.6                                                  
    h["GUSTD"]   =295.6                                                  
    h["GUSTS"]   =25.1                                                  
    h["AIRTEMP2"]=8.1                                                  
    h["DEWPOINT"]=-4.2                                                  
    h["TRUSTEMP"]=7.79                                                  
    h["HUMIDITY"]=39.9                                                  
    h["DUSTA"]   =16084.0                                                  
    h["DUSTB"]   =1020.0                                                  
    h["WINDD25M"]=318.3                                                  
    h["WINDS25M"]=1.4    
    if 'flat    ' in flab:                                              
        h["FF"]      = ('1 1 1 1 ','FF lamps 1:on 0:0ff')                       
        h["NE"]      = ('0 0 0 0 ','NE lamps 1:on 0:0ff')                          
        h["HGCD"]    = ('0 0 0 0 ','HGCD lamps 1:on 0:0ff')                          
        h["FFS"]     = ('1 1 1 1 1 1 1 1','Flatfield Screen 1:closed 0:open')    
    elif 'science ' in flab:
        h["FF"]      = ('0 0 0 0 ','FF lamps 1:on 0:0ff')                       
        h["NE"]      = ('0 0 0 0 ','NE lamps 1:on 0:0ff')                          
        h["HGCD"]    = ('0 0 0 0 ','HGCD lamps 1:on 0:0ff')                          
        h["FFS"]     = ('0 0 0 0 0 0 0 0','Flatfield Screen 1:closed 0:open')                     
    elif 'arc     ' in flab:
        h["FF"]      = ('0 0 0 0 ','FF lamps 1:on 0:0ff')                       
        h["NE"]      = ('1 1 1 1 ','NE lamps 1:on 0:0ff')                          
        h["HGCD"]    = ('1 1 1 1 ','HGCD lamps 1:on 0:0ff')                          
        h["FFS"]     = ('1 1 1 1 1 1 1 1','Flatfield Screen 1:closed 0:open') 
    h["MGDPOS"]  = ('C       ','MaNGA dither position (C,N,S,E)')             
    h["MGDRA"]   = (0.0,'MaNGA decenter in RA, redundant with MGDPOS') 
    h["MGDDEC"]  = (0.0,'MaNGA decenter in Dec, redundant with MGDPOS')  
    h["GUIDER1"] = ('proc-gimg-0500.fits.gz','The first guider image')              
    h["SLITID1"] = (16,'Normalized slithead ID. sp1&2 should match.') 
    h["SLITID2"] = (16,'Normalized slithead ID. sp1&2 should match.')  
    h["GUIDERN"] = ('proc-gimg-0529.fits.gz','The last guider image')  
    h["COLLA"]   = (1173,'The position of the A collimator motor')         
    h["COLLB"]   = (164,'The position of the B collimator motor')       
    h["COLLC"]   = (577,'The position of the C collimator motor')       
    h["HARTMANN"]= ('Out     ','Hartmanns: Left,Right,Out')   
    if '2' in typ:    
        h["MC2HUMHT"]= (32.3,'sp2 mech Hartmann humidity, %')                  
        h["MC2HUMCO"]= (25.6,'sp2 mech Central optics humidity, %')           
        h["MC2TEMDN"]= (7.2,'sp2 mech Median temp, C')                
        h["MC2THT"]  = (7.5,'sp2 mech Hartmann Top Temp, C')          
        h["MC2TRCB"] = (7.4,'sp2 mech Red Cam Bottom Temp, C')                
        h["MC2TRCT"] = (6.9,'sp2 mech Red Cam Top Temp, C')              
        h["MC2TBCB"] = (7.1,'sp2 mech Blue Cam Bottom Temp, C')               
        h["MC2TBCT"] = (7.2,'sp2 mech Blue Cam Top Temp, C')   
    else:
        h["MC2HUMHT"]= (32.3,'sp1 mech Hartmann humidity, %')                  
        h["MC2HUMCO"]= (25.6,'sp1 mech Central optics humidity, %')           
        h["MC2TEMDN"]= (7.2,'sp1 mech Median temp, C')                
        h["MC2THT"]  = (7.5,'sp1 mech Hartmann Top Temp, C')          
        h["MC2TRCB"] = (7.4,'sp1 mech Red Cam Bottom Temp, C')                
        h["MC2TRCT"] = (6.9,'sp1 mech Red Cam Top Temp, C')              
        h["MC2TBCB"] = (7.1,'sp1 mech Blue Cam Bottom Temp, C')               
        h["MC2TBCT"] = (7.2,'sp1 mech Blue Cam Top Temp, C')                 
    h["REQTIME"] = (expt,'requested exposure time')             
    h["EXPTIME"] = (expt+0.14,'measured exposure time, s')                      
    h["SHOPETIM"]= (0.6899999999999999,'open shutter transit time, s')                   
    h["SHCLOTIM"]= (0.63,'close shutter transit time, s')                  
    h["DARKTIME"]= (expt+9.4519929885864,'time between flush end and readout start')       
    h["LN2TEMP"] = 81.64100000000001                                                  
    h["CCDTEMP"] = -133.984                                                  
    h["IONPUMP"] = -6.17                                                  
    h["BSCALE"]  = 1                                                  
    h["BZERO"]   = 32768                                                  
    h["CHECKSUM"]= ('DrANEo1KDo8KDo8K','HDU checksum updated 2016-05-10T06:58:02')   
    h["DATASUM"] = ('516485492','data unit checksum updated 2016-05-10T06:58:02')                                                                          
    return h

    
def run_2d(blu_sp1,fibf,base_name='test',dir1='',nfib=500,type="blue",cam=1,expN=0,expt=900,ra=0,dec=0,mjd='45223',field_name='00000'):
    expof=0
    raw_exp_bhm(blu_sp1,fibf,base_name,fc=[0.88,0.94],n_cr=600,cam=cam,nfib=nfib,type=type,dir1=dir1,mjd=mjd,plate=field_name,flb='s',exp=expN,expt=expt,ra0=ra,dec0=dec,expof=expof)
    
