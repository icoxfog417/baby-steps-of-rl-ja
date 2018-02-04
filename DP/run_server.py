import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import tornado.ioloop
from tornado.options import define, options, parse_command_line
from DP.application import Application


define("port", default=8888, help="run on the given port", type=int)


def main():
    parse_command_line()
    app = Application()
    app.listen(options.port)
    print("Run server on port: {}".format(options.port))
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
