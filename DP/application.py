import os
import tornado.web
import tornado.escape
from environment import Environment
from planner import ValueIterationPlanner, PolicyIterationPlanner


class IndexHandler(tornado.web.RequestHandler):

    def get(self):
        self.render("index.html")


class PlanningHandler(tornado.web.RequestHandler):

    def post(self):
        data = tornado.escape.json_decode(self.request.body) 
        grid = data["grid"]
        plan_type = data["plan"]
        move_prob = 0.8  # default value

        try:
            move_prob = float(data["prob"])
        except ValueError:
            pass

        env = Environment(grid, move_prob=move_prob)
        if plan_type == "value":
            planner = ValueIterationPlanner(env)
        elif plan_type == "policy":
            planner = PolicyIterationPlanner(env)

        result = planner.plan()
        planner.log.append(result)
        self.write({"log": planner.log})


class Application(tornado.web.Application):

    def __init__(self):
        handlers = [
            (r"/", IndexHandler),
            (r"/plan", PlanningHandler),
        ]

        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            cookie_secret=os.environ.get("SECRET_TOKEN", "__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__"),
            debug=True,
        )

        super(Application, self).__init__(handlers, **settings)
