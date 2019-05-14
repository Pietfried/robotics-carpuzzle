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
        while (wait and not rck.is_idle()):
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


if __name__ == '__main__':
    rck = RemoteControlKUKA()
    open_grp = True
    default_Axis = ',A -175.18,B 0,C -180,S 6,T 27,E1 0.0,E2 0.0,E3 0.0,E4 0.0,E5 0.0,E6 0.0}'

    while True:
        time.sleep(1)
        rck.open_grp(open_grp)
        open_grp = not open_grp
        rck.move_lin_e6pos('{X 0,Y 0,Z 0' + default_Axis)

