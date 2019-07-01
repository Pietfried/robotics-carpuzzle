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
    """
    Defines a class for controlling the kuka robot using the RemoteControllKuka class.
    This class provides public methods for picking and placing puzzle pieces and for going into resting position.
    Also it contains a method for calculating the offset between the KUKA coordinates and the coordinates provided
    by the image processing
    """

    def __init__(self):
        self.kukaRemote = RemoteControlKUKA()
        #Default hights in mm above the foam
        self.z_Pick = -10
        self.z_Travel = -60
        self.z_Place = -25
        self.z_Default = -185
        self.z_Draw = -60
        self.z_Draw_Travel = -185

        #Movements to the side using shaking (pixel)
        self.xy_shake = 12

        #Conversion factor from pixels to mm
        self.convertCoordiante = 1 / (2.3022)

        #Offset in pixel (x, y)
        self.offset = (10.11, -8.51)

        #Needed for the RemoteControlKUKA class, but not relevant for this project
        self.DEFAULT_B = 0
        self.DEFAULT_C = 0

        #Define resting position
        self.defaultAngle = 0
        self.defaultPosition = (0, 0)

        self.currentPosition = self.defaultPosition

    def __createKUKA_CMD(self, xy, z, angle):
        """

        :param xy: tuple of coordinates (pixel)
        :param z: height in mm
        :param angle: angle in degree
        :return: string for communication with RemoteControlKUKA class
        """
        xy_offset= (xy[0]+self.offset[0], xy[1]+self.offset[1])
        strCoord = '{X ' + str(xy_offset[0] * self.convertCoordiante) + ',Y ' + str(
            xy_offset[1] * self.convertCoordiante) + ',Z ' + str(z)
        strAngle = ',A ' + str(angle) + ' ,B ' + str(self.DEFAULT_B) + ',C ' + str(self.DEFAULT_C) + '}'
        return strCoord + strAngle

    def __go2Position(self, xy, z, angle, ptp=False):
        """
        Steering the kuka into a specific location (predefines workspace is set to outside-stop, so violation
        of workspace must not considerd here)

        :param xy: tuples of coordiantes (pixel)
        :param z: height in mm
        :param angle: angle in degree
        :param ptp: True - point-to-point movemtent is executed
        :return: none
        """
        position = self.__createKUKA_CMD(xy, z, angle)
        if ptp:
            self.kukaRemote.move_ptp_e6pos(position)
        else:
            self.kukaRemote.move_lin_e6pos(position)

        while not self.kukaRemote.is_idle():
            pass
        self.currentPosition = xy

    def pick(self, xy):
        """

        :param xy: tuple of coordinates (pixel)
        :return: none
        """
        self.__go2Position(self.currentPosition, self.z_Travel, self.defaultAngle)
        self.__go2Position(xy, self.z_Travel, self.defaultAngle)
        self.kukaRemote.open_grp(False)
        self.__go2Position(xy, self.z_Pick, self.defaultAngle)
        self.kukaRemote.open_grp(True)
        self.__go2Position(xy, self.z_Travel, self.defaultAngle)
        pass

    def shake_X(self, xy, angle):
        """
        Shakes the tool-center-point around a given center point in X direction (First in positive x direction,
        followed by movment in negative x diraction, and same principle for y direction)

        :param xy: center position for shaking (pixel)
        :param angle: angle in degree (just needed to prevent the reset of axis 6 while shaking)
        :return: none
        """
        self.__go2Position((xy[0] + self.xy_shake, xy[1]), self.z_Place, angle)
        self.__go2Position((xy[0] - self.xy_shake, xy[1]), self.z_Place, angle)
        self.__go2Position(xy, self.z_Place, angle)
        self.__go2Position((xy[0], xy[1] + self.xy_shake), self.z_Place, angle)
        self.__go2Position((xy[0], xy[1] - self.xy_shake), self.z_Place, angle)
        self.__go2Position(xy, self.z_Place, angle)

    def shake_O(self,xy,angle):
        """
        Shakes the tool-center-point around a given center point in O direction (First in positive x direction,
        followed by movment in positive y direction -> positive y diecrion and negative x...)

        :param xy: center position for shaking (pixel)
        :param angle: angle in degree (just needed to prevent the reset of axis 6 while shaking)
        :return: none
        """

        self.__go2Position((xy[0] + self.xy_shake, xy[1]), self.z_Place, angle)
        self.__go2Position((xy[0], xy[1] + self.xy_shake), self.z_Place, angle)
        self.__go2Position((xy[0] - self.xy_shake, xy[1]), self.z_Place, angle)
        self.__go2Position((xy[0], xy[1] - self.xy_shake), self.z_Place, angle)
        self.__go2Position(xy, self.z_Place, angle)

    def place(self, xy, angle, standardUse = True, doShaking='X'):
        """
        Steer KUKA to a coordinate to place a puzzle piece inside the puzzle

        :param xy: coordinates of the puzzle slot center (pixel)
        :param angle: rotation of the piece (degree)
        :param standardUse: True - standard place a puzzle piece /False: used for offset method
                            (dont drop the pen, and don't go to deep)
        :param doShaking: 'X' for shaking with X principle, 'O' for shaking in O principle, and anything esle for not doing shaking
        :return: none
        """
        if(standardUse):
            self.__go2Position(xy, self.z_Travel, angle)
            self.__go2Position(xy, self.z_Place, angle)
            self.kukaRemote.open_grp(False)

            #Shake puzzle piece inside the board
            if(doShaking=='X'):
                self.shake_X(xy,angle)
            elif(doShaking=='O'):
                self.shake_O(xy,angle)

            self.__go2Position(xy, self.z_Travel, self.defaultAngle)
        else:
            self.__go2Position(xy, self.z_Draw_Travel, angle)
            self.__go2Position(xy, self.z_Draw, angle)
            self.__go2Position(xy, self.z_Draw_Travel, angle)

        pass

    def go2Origin(self):
        """
        Reset the kuka to origin / resting position
        :return: none
        """
        self.kukaRemote.open_grp(False)
        self.__go2Position(self.defaultPosition, self.z_Default, self.defaultAngle, ptp=False)
        pass

    def convert_angle(self,angle):
        """
        Convert angle from image processing to correct angle for placement (only turn at most 180°) and correct left and right rotation
        :param angle: angle in degree
        :return: none
        """
        if (abs(angle) > 180):
            angle = (-angle % 180)
        else:
            angle *= -1

        return angle

    def draw_Points4Offset(self):
        """
        Draw points into the workspace so that the image processing can calculate the offset
        :return: none
        """
        self.kukaRemote.open_grp(True)
        for x in range(50,1000,100):
            for y in range(50,700,100):
                self.place(xy=(x,y), angle=0, standardUse=False)

        kuka.go2Origin()

if __name__ == '__main__':
    #Just used to draw points into the workspace to calculate the current offset
    kuka = PuzzleSolver()
    kuka.go2Origin()
    kuka.draw_Points4Offset()
