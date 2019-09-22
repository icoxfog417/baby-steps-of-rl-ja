import os
import tornado.ioloop
from tornado.options import define, options, parse_command_line
from application import Application


define("port", default=8888, help="run on the given port", type=int)


def main():
    parse_command_line()
    app = Application()
    port = int(os.environ.get("PORT", 8888))
    app.listen(port)
    print("Run server on port: {}".format(port))
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()
