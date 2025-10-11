# robot_arm.py (251011 튜닝 적용)
import numpy as np

class RobotArm:
    def __init__(self):
        self.base=np.array([0.0,0.0,20.0])
        self.L1,self.L2,self.L3=60.0,80.0,70.0
        self.joints=[0,0,0,0,0]
        self.end_effector=np.array([0.0,0.0,0.0])
        self.servo_map={1:(-90,90,0,180),2:(-90,10,120,20),3:(0,140,30,170),4:(-90,90,0,180),5:(-90,90,0,180)}

    def forward_kinematics(self):
        j1,j2,j3,j4,j5=np.radians(self.joints)
        x0,y0,z0=self.base
        x1=x0;y1=y0;z1=z0+self.L1*np.cos(j2)
        x2=x1+self.L2*np.cos(j2+j3)*np.cos(j1)
        y2=y1+self.L2*np.cos(j2+j3)*np.sin(j1)
        z2=z1+self.L2*np.sin(j2+j3)
        x3=x2+self.L3*np.cos(j2+j3+j4)*np.cos(j1)
        y3=y2+self.L3*np.cos(j2+j3+j4)*np.sin(j1)
        z3=z2+self.L3*np.sin(j2+j3+j4)
        self.end_effector=np.array([x3,y3,z3])
        return [self.base,[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]

    def inverse_kinematics(self,x,y,z):
        dx,dy,dz=x-self.base[0],y-self.base[1],z-self.base[2]
        dist=np.sqrt(dx**2+dy**2+dz**2); reach=self.L1+self.L2+self.L3-5
        if dist>reach:
            print("[WARN] Target out of reach."); return self.joints
        j1=np.degrees(np.arctan2(dy,dx)); r=np.sqrt(dx**2+dy**2); h=dz; d=np.sqrt(r**2+h**2)
        d=np.clip(d,1e-6,reach)
        v2=(self.L1**2+d**2-self.L2**2)/(2*self.L1*d); v3=(self.L1**2+self.L2**2-d**2)/(2*self.L1*self.L2)
        v2,v3=np.clip(v2,-1,1),np.clip(v3,-1,1)
        a1=np.arctan2(h,r); a2=np.arccos(v2); a3=np.arccos(v3)
        j2=np.degrees(a1+a2); j3=np.degrees(np.pi-a3)
        j4=-(j2+j3-90); j5=0
        self.joints=[j1,j2-90,-(j3-90),j4,j5]; self.update_end_effector()
        return self.joints

    def update_end_effector(self):
        pts=self.forward_kinematics(); self.end_effector=np.array(pts[-1])

    def get_servo_angles(self):
        out=[]
        table={0:(-90,90,0,180),1:(-90,10,120,20),2:(0,140,30,170),3:(-90,90,0,180),4:(-90,90,0,180)}
        for i,v in enumerate(self.joints):
            sim_min,sim_max,real_min,real_max=table[i]
            v=np.clip(v,sim_min,sim_max)
            m=np.interp(v,[sim_min,sim_max],[real_min,real_max])
            out.append(int(m))
        return out
