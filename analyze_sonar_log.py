import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

sonar_t,sonar_v = np.loadtxt('/root/lake_ping.csv',dtype=str,delimiter=',',usecols=(0, 5),unpack=True)
gps_t,gps_x, gps_y = np.loadtxt('/root/lake_gps.csv',dtype=str,delimiter=',',usecols=(0, 7,6),unpack=True)

gps_t_vec = gps_t[1:5600]
gps_t_vec = gps_t_vec.astype(np.float)

sonar_t_vec = sonar_t[1:11500]
sonar_t_vec = sonar_t_vec.astype(np.float)
sonar_v_vec = sonar_v[1:11500]
sonar_t_vec = sonar_t[1:11500]
sonar_t_vec = sonar_t_vec.astype(np.float)
sonar_v_vec = sonar_v_vec.astype(np.float)
sonar_v_itrp = np.interp(gps_t_vec, sonar_t_vec, sonar_v_vec)

gpsx = gps_x[1:5600]
gpsy = gps_y[1:5600]
gpsy = gpsy.astype(np.float)
gpsx = gpsx.astype(np.float)
plt.scatter(gpsx,gpsy,s=20,c=sonar_v_itrp/1000.0,marker='o',cmap=cm.jet,lw=0)
cbar = plt.colorbar()
cbar.solids.set_edgecolor("face")
cbar.ax.set_ylabel('Depth (meter)', rotation=270)
plt.ylabel('Latitude')
plt.xlabel('Longitude')
plt.draw()
plt.show()
