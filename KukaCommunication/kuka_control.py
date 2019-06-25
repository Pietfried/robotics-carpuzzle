import time
import cv2
from KukaCommunication.py_openshowvar import openshowvar
from image_processing import puzzle_image_processing as imgp

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

    def open_grp(self, open):
        # green valve closed with out2
        # red valve closed with out1
        # out2 = 1 -> green valve closed
        # out1 = 1 -> red valve opened

        if (open == True):
            self._set_green_Valve("FALSE")
            self._set_red_Valve("TRUE")
        else:
            self._set_green_Valve("TRUE")
            self._set_red_Valve("FALSE")

        self._set_action("Gripper")


class PuzzleSolver:
    def __init__(self):
        self.kukaRemote = RemoteControlKUKA()

        self.z_Pick = -10#-150#-7
        self.z_Travel = -60#-150
        self.z_Place = -25#-59.5
        self.z_Default = -185#-280

        self.convertCoordiante = 1 / (2.3022)
        self.offset = (5.32, -10.4)

        self.DEFAULT_B = 0
        self.DEFAULT_C = 0

        self.defaultAngle = 0
        self.defaultPosition = (0, 0)

        self.currentPosition = self.defaultPosition

    def __createKUKA_CMD(self, xy, z, angle):
        xy_offset= (xy[0]+self.offset[0], xy[1]+self.offset[1])
        strCoord = '{X ' + str(xy_offset[0] * self.convertCoordiante) + ',Y ' + str(
            xy_offset[1] * self.convertCoordiante) + ',Z ' + str(z)
        strAngle = ',A ' + str(angle) + ' ,B ' + str(self.DEFAULT_B) + ',C ' + str(self.DEFAULT_C) + '}'
        return strCoord + strAngle

    def __go2Position(self, xy, z, angle, ptp=False):
        position = self.__createKUKA_CMD(xy, z, angle)
        if ptp:
            self.kukaRemote.move_ptp_e6pos(position)
        else:
            self.kukaRemote.move_lin_e6pos(position)

        while not self.kukaRemote.is_idle():
            pass
        self.currentPosition = xy

    def pick(self, xy):
        self.__go2Position(self.currentPosition, self.z_Travel, self.defaultAngle)
        self.__go2Position(xy, self.z_Travel, self.defaultAngle)
        self.kukaRemote.open_grp(False)
        self.__go2Position(xy, self.z_Pick, self.defaultAngle)
        self.kukaRemote.open_grp(True)
        self.__go2Position(xy, self.z_Travel, self.defaultAngle)
        pass

    def place(self, xy, angle):
        self.__go2Position(xy, self.z_Travel, angle)
        self.__go2Position(xy, self.z_Place, angle)
        self.kukaRemote.open_grp(False)
        self.__go2Position(xy, self.z_Travel, self.defaultAngle)
        pass

    def go2Origin(self):
        self.kukaRemote.open_grp(False)
        self.__go2Position(self.defaultPosition, self.z_Default, self.defaultAngle, ptp=False)
        pass

    def convert_angle(self,angle):

        if (abs(angle) > 180):
            angle = (-angle % 180)
        else:
            angle *= -1

        return angle

if __name__ == '__main__':
    kuka = PuzzleSolver()
    kuka.go2Origin()

    img = cv2.imread('C:/Users/CarPuzzle/Desktop/git repository/image_processing/images/image39.jpg')
    puzzlepieces, slots = imgp.init_pieces_and_slots(img)

    #kuka.pick(xy=puzzlepieces[4].handle_center)
    #kuka.place(xy=puzzlepieces[4].match.center, angle= kuka.convert_angle(puzzlepieces[4].angle))

    #kuka.go2Origin()

    for piece in puzzlepieces:
        kuka.pick(xy=piece.handle_center)
        kuka.place(xy=piece.match.center, angle= kuka.convert_angle(piece.angle))

    kuka.go2Origin()

    #kuka.pick(xy=(431, 703), angle=0)
    #kuka.pick(xy=(,653), angle=0)
    #input()

    # kuka.pick(xy=(520,393), angle=0)
    #input()

    # kuka.pick(xy=(93,109), angle=0)
    #input()

    # kuka.pick(xy=(935,94), angle=0)
    #input()

    # kuka.place(xy=(905, 372), angle=-58.2939)

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
