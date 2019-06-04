import time
from py_openshowvar import openshowvar


class RemoteControlKUKA:

    def __init__(self, address="192.168.1.15", port=7000, debug=False):
        self.debug = debug
        self._client = openshowvar(address, port)
        if (not self._client.can_connect):
            raise RuntimeError("Unable to connect.")

    def __del__(self):
        self._client.close()

    def _read(self, identifier, debug=False):
        return self._client.read(identifier, debug=debug)

    def _write(self, identifier, value, debug=False):
        return self._client.write(identifier, value, debug=self.debug or debug)

    def get_pos(self):
        return self._read("$POS_ACT")

    def is_idle(self):
        return b'0' == self._read("COM_ACTION")

    def _set_e6pos(self, e6pos):
        self._write("COM_E6POS", e6pos)

    def _set_red_Valve(self, value):
        self._write("COM_RED_VALVE", value)

    def _set_green_Valve(self, value):
        self._write("COM_GREEN_VALVE", value)

    def _set_action(self, s_action):
        if (s_action == "PTP_E6POS"):
            a = 11
        elif (s_action == "LIN_E6POS"):
            a = 12
        elif (s_action == "Gripper"):
            a = 13
        else:
            raise NotImplementedError("Unknown action string.")
        self._write("COM_ACTION", str(a))

    def _wait(self, wait=True):
        while (wait and not self.is_idle()):
            time.sleep(0.1)

    def move_lin_e6pos(self, e6pos, block=True):
        self._set_e6pos(e6pos)
        self._set_action("LIN_E6POS")
        self._wait(block)

    def move_ptp_e6pos(self, e6pos, block=True):
        self._set_e6pos(e6pos)
        self._set_action("PTP_E6POS")
        self._wait(block)

    def open_grp(self,open):
        # green valve closed with out2
        # red valve closed with out1
        # out2 = 1 -> green valve closed
        # out1 = 1 -> green valve opened

        if(open == True):
            self._set_green_Valve("FALSE")
            self._set_red_Valve("TRUE")
        else:
            self._set_green_Valve("TRUE")
            self._set_red_Valve("FALSE")

        self._set_action("Gripper")

class PuzzleSolver:
    def __init__(self):
        self.kukaRemote = RemoteControlKUKA()

        self.z_Pick = -27
        self.z_Travel = -40
        self.z_Place = -15
        self.z_Default = -179.5

        self.convertCoordiante = 1/(2.3022)

        self.DEFAULT_B = 0
        self.DEFAULT_C = 0

        self.defaultAngle = 0
        self.defaultPosition = (0,0)
        self.currentPosition = self.defaultPosition[0]

    def __createKUKA_CMD(self, xy, z, angle):
        strCoord = '{X ' + str(xy[0] * self.convertCoordiante) + ',Y ' + str(xy[1] * self.convertCoordiante) + ',Z ' + str(z)
        strAngle = ',A ' + str(angle) + ' ,B ' + str(self.DEFAULT_B) + ',C ' + str(self.DEFAULT_C) + '}'
        return strCoord + strAngle

    def __go2Position(self,xy,z,angle):
        position = self.__createKUKA_CMD(xy,z,angle)
        self.kukaRemote.move_lin_e6pos(position)
        while not self.kukaRemote.is_idle():
            pass
        self.currentPosition = xy

    def pick(self, xy, angle=0):
        self.__go2Position(self.currentPosition, self.z_Travel, self.defaultAngle)
        self.__go2Position(xy, self.z_Travel, self.defaultAngle)
        self.kukaRemote.open_grp(False)
        self.__go2Position(xy, self.z_Pick, self.defaultAngle)
        #self.kukaRemote.open_grp(True)
        #self.__go2Position(xy, self.z_Travel, angle)
        pass

    def place(self, xy,angle):
        self.__go2Position(xy, self.z_Travel, angle)
        self.__go2Position(xy, self.z_Place, angle)
        self.kukaRemote.open_grp(False)
        self.__go2Position(xy, self.z_Travel, self.defaultAngle)
        pass

    def go2Origin(self):
        self.kukaRemote.open_grp(False)
        self.__go2Position(self.defaultPosition, self.z_Default, self.defaultAngle)
        pass



if __name__ == '__main__':
    kuka = PuzzleSolver()
    kuka.go2Origin()
    kuka.kukaRemote.open_grp(True)
    #kuka.pick(xy=(923, 710), angle=0)
    input()

    #kuka.pick(xy=(91,653), angle=0)
    input()

    #kuka.pick(xy=(520,393), angle=0)
    input()

    #kuka.pick(xy=(93,109), angle=0)
    input()

    #kuka.pick(xy=(935,94), angle=0)
    input()

    #kuka.place(xy=(905, 372), angle=-58.2939)

#    rck = RemoteControlKUKA()
#    open_grp = True
#    default_Axis = ',A -175.18,B 0,C -180,S 6,T 27,E1 0.0,E2 0.0,E3 0.0,E4 0.0,E5 0.0,E6 0.0}'####

#    rck.move_lin_e6pos('{X 340,Y 225,Z -10,A -10,B 0,C 0}')
#    rck.open_grp(True)
#    rck.move_lin_e6pos('{X 340,Y 225,Z -40,A -10,B 0,C 0}')
#    rck.move_lin_e6pos('{X 50.1234,Y 50.1234,Z -40,A -10,B 0,C 0}')
#    rck.move_lin_e6pos('{X 50.1234,Y 50.1234,Z -7,A 0,B 0,C 0}')
#    rck.open_grp(False)
#    rck.move_lin_e6pos('{X 50.1234,Y 50.1234,Z -40,A 0,B 0,C 0}')
#    time.sleep(1)
#    rck.move_lin_e6pos('{X 50.1234,Y 50.1234,Z -7,A 0,B 0,C 0}')
#    rck.open_grp(True)
#    rck.move_lin_e6pos('{X 50.1234,Y 50.1234,Z -40,A 0,B 0,C 0}')
#    rck.move_lin_e6pos('{X 340,Y 225,Z -40,A -10,B 0,C 0}')
#    rck.move_lin_e6pos('{X 340,Y 225,Z -15,A -10,B 0,C 0}')
#    rck.open_grp(False)

