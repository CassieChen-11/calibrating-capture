import sys
sys.path.append(r"E:\MVviewer\MV Viewer\Development\Samples\Python\IMV\MVSDK")
from IMVApi import *

deviceList = IMV_DeviceList()
MvCamera.IMV_EnumDevices(deviceList, IMV_EInterfaceType.interfaceTypeAll)

cam = MvCamera()
idx = c_void_p(0)
cam.IMV_CreateHandle(IMV_ECreateHandleMode.modeByIndex, byref(idx))
cam.IMV_Open()

# 尝试列出所有属性
try:
    feature_list = IMV_FeatureList()
    nRet = cam.IMV_GetFeatureList(feature_list)
    if nRet == IMV_OK:
        print(f"相机支持 {feature_list.nFeatureNum} 个属性：")
        for i in range(feature_list.nFeatureNum):
            name = feature_list.pFeatureInfo[i].FeatureName.decode("utf-8")
            if "Exposure" in name or "Gain" in name:
                print(f"  -> {name}")
    else:
        print(f"获取属性列表失败, 错误码={nRet}")
except Exception as e:
    print(f"异常: {e}")

cam.IMV_Close()
cam.IMV_DestroyHandle()