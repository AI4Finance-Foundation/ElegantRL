import math

import Box2D
import gym
import numpy as np
import numpy.random as rd
import pyglet
from gym.envs.box2d.car_dynamics import Car
from pyglet import gl  # must gym<=0.17.1, pyglet==1.5.0

# Easiest continuous control task to learn from pixels, a top-down racing environment.
# Discrete control is reasonable in this environment as well, on/off discretization is
# fine.
#
# State consists of STATE_W x STATE_H pixels.
#
# Reward is -0.1 every frame and +1000/N for every track tile visited, where N is
# the total number of tiles visited in the track. For example, if you have finished in 732 frames,
# your reward is 1000 - 0.1*732 = 926.8 points.
#
# Game is solved when agent consistently gets 900+ points. Track generated is random every episode.
#
# Episode finishes when all tiles are visited. Car also can go outside of PLAYFIELD, that
# is far off the track, then it will get -100 and die.
#
# Some indicators shown at the bottom of the window and the state RGB buffer. From
# left to right: true speed, four ABS sensors, steering wheel position and gyroscope.
#
# To play yourself (it's rather fast for humans), type:
#
# python gym/envs/box2d/car_racing.py
#
# Remember it's powerful rear-wheel drive car, don't press accelerator and turn at the
# same time.
#
# Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
# Modify by Github User Yonv1943 (2021-08-08)

STATE_W = 112  # Yonv1943  # STATE_H = 96  # less than Atari 160x192
STATE_H = 112  # Yonv1943  # STATE_H = 96
WINDOW_W = 300
WINDOW_H = 300

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 1.1  # Yonv1943 # ZOOM = 2.7  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]


class CarRacingFix:
    assert gym.__version__ <= "0.17.1"

    def __init__(self, verbose=1):
        self.contactListener_keep_ref = FrictionDetector(self)
        self.world = Box2D.b2World(
            (0, 0), contactListener=self.contactListener_keep_ref
        )
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose

        self.fd_tile = Box2D.b2FixtureDef(
            shape=Box2D.b2PolygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        self.state_temp = None  # Yonv1943
        self.tile_visited_count = 0
        self.road_poly = []
        self.transform = None
        self.t = None
        self.num_step = 0

        self.env_name = "CarRacingFix"
        self.state_dim = (STATE_W, STATE_H, 3 * 2)
        self.action_dim = 6
        self.if_discrete = False
        self.max_step = 512
        self.target_return = 950
        self.action_max = 1

    def reset(self):
        self.num_step = 1

        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []

        while True:
            success = self._create_track()
            if success:
                break
            # if self.verbose == 1:
            #     print("retry to generate track (normal if there are not many of this messages)")
        self.car = Car(self.world, *self.track[0][1:4])

        self.state_temp = np.zeros((STATE_W, STATE_H, 3), dtype=np.uint8)
        return self.old_step((0, 0, 0))[0]

    def step(self, action):
        try:
            reward0 = self.old_step(action[:3], if_draw=False)[1]
            state, reward1, done, info_dict = self.old_step(action[3:], if_draw=True)
            reward = reward0 + reward1
        except Exception as error:
            print(f"| CarRacingFix Error: {error}")
            state = np.stack((self.state_temp, self.state_temp))
            reward = 0
            done = True
            info_dict = {}
        self.num_step += 1
        if self.num_step == self.max_step:
            done = True
        return state, reward, done, info_dict

    def old_step(self, action, if_draw=True):
        self.car.steer(action[0])
        self.car.gas(action[1])  # np.clip(gas, 0, 1)
        self.car.brake(action[2])

        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.reward -= 0.1
        # We actually don't want to count fuel spent, we want car to be faster.
        # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
        # self.car.fuel_spent = 0.0

        done = False
        # x, y = self.car.hull.position
        # if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
        #     done = True
        #     step_reward = -100  # Ynv1943: it is a bad design
        if if_draw:
            state = self.render("state_pixels")
            if not (
                32 < state[16:96, 16:96, 1].mean() < 211
            ):  # penalize when outside of road
                # print(f"{state[16:96, 16:96, 1].mean():.3f}")
                self.reward -= 2.0
                done = True
            if self.tile_visited_count == len(self.track):
                done = True
            stack_state = np.concatenate((self.state_temp, state), axis=2)
            self.state_temp = state
        else:
            stack_state = None
        step_reward = self.reward - self.prev_reward
        self.prev_reward = self.reward

        return stack_state, step_reward, done, {}

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render(self, mode="human"):
        assert mode in ["human", "state_pixels"]
        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            # self.score_label = pyglet.text.Label('0000', font_size=36,
            #                                      x=20, y=WINDOW_H * 2.5 / 40.00, anchor_x='left', anchor_y='center',
            #                                      color=(255, 255, 255, 255))
            self.transform = rendering.Transform()
        if self.t is None:
            return None

        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(
            self.t, 1
        )  # Animate zoom first second
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2
            - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4
            - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)),
        )
        self.transform.set_rotation(angle)

        self.car.draw(self.viewer, mode != "state_pixels")

        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "state_pixels":
            vp_w = STATE_W
            vp_h = STATE_H
        else:
            context_nscontext = getattr(win.context, "_nscontext", None)
            pixel_scale = (
                1
                if context_nscontext is None
                else context_nscontext.view().backingScaleFactor()
            )
            # pylint: disable=protected-access
            vp_w = int(pixel_scale * WINDOW_W)
            vp_h = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, vp_w, vp_h)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        # self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape((vp_h, vp_w, 4))[:, :, :3]
        return arr

    def render_road(self):
        gl.glBegin(gl.GL_QUADS)
        gl.glColor4f(0.4, 0.8, 0.4, 1.0)
        gl.glVertex3f(-PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, +PLAYFIELD, 0)
        gl.glVertex3f(+PLAYFIELD, -PLAYFIELD, 0)
        gl.glVertex3f(-PLAYFIELD, -PLAYFIELD, 0)
        gl.glColor4f(0.4, 0.9, 0.4, 1.0)
        k = PLAYFIELD / 20.0
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                gl.glVertex3f(k * x + k, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + 0, 0)
                gl.glVertex3f(k * x + 0, k * y + k, 0)
                gl.glVertex3f(k * x + k, k * y + k, 0)
        for poly, color in self.road_poly:
            gl.glColor4f(color[0], color[1], color[2], 1)
            for p in poly:
                gl.glVertex3f(p[0], p[1], 0)
        gl.glEnd()

    def render_indicators(self, w, h):
        gl.glBegin(gl.GL_QUADS)
        s = w / 40.0
        h = h / 40.0
        gl.glColor4f(0, 0, 0, 1)
        gl.glVertex3f(w, 0, 0)
        gl.glVertex3f(w, 5 * h, 0)
        gl.glVertex3f(0, 5 * h, 0)
        gl.glVertex3f(0, 0, 0)

        def vertical_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h + h * val, 0)
            gl.glVertex3f((place + 1) * s, h, 0)
            gl.glVertex3f((place + 0) * s, h, 0)

        def horiz_ind(place, val, color):
            gl.glColor4f(color[0], color[1], color[2], 1)
            gl.glVertex3f((place + 0) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 4 * h, 0)
            gl.glVertex3f((place + val) * s, 2 * h, 0)
            gl.glVertex3f((place + 0) * s, 2 * h, 0)

        true_speed = np.sqrt(
            np.square(self.car.hull.linearVelocity[0])
            + np.square(self.car.hull.linearVelocity[1])
        )
        vertical_ind(5, 0.02 * true_speed, (1, 1, 1))
        vertical_ind(7, 0.01 * self.car.wheels[0].omega, (0.0, 0, 1))  # ABS sensors
        vertical_ind(8, 0.01 * self.car.wheels[1].omega, (0.0, 0, 1))
        vertical_ind(9, 0.01 * self.car.wheels[2].omega, (0.2, 0, 1))
        vertical_ind(10, 0.01 * self.car.wheels[3].omega, (0.2, 0, 1))
        horiz_ind(20, -10.0 * self.car.wheels[0].joint.angle, (0, 1, 0))
        horiz_ind(30, -0.8 * self.car.hull.angularVelocity, (1, 0, 0))
        gl.glEnd()
        # self.score_label.text = "%04i" % self.reward
        # self.score_label.draw()

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self):
        check_point = 12

        # Create checkpoints
        checkpoints = []
        for c in range(check_point):
            alpha = 2 * math.pi * c / check_point + rd.uniform(
                0, 2 * math.pi * 1 / check_point
            )
            rad = rd.uniform(TRACK_RAD / 3, TRACK_RAD)
            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == check_point - 1:
                alpha = 2 * math.pi * c / check_point
                self.start_alpha = 2 * math.pi * (-0.5) / check_point
                rad = 1.5 * TRACK_RAD
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))

        # print "\n".join(str(h) for h in checkpoints)
        # self.road_poly = [ (    # uncomment this to see checkpoints
        #    [ (tx,ty) for a,tx,ty in checkpoints ],
        #    (0.7,0.7,0.9) ) ]
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi
            while True:  # Find destination from checkpoints
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2 * math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy  # destination vector projected on rad
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break
        # print "\n".join([str(t) for t in enumerate(track)])

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = track[i][0] > self.start_alpha >= track[i - 1][0]
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        # if self.verbose == 1:  # Yonv1943
        #     print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            t.road_friction = 1.0
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0))
                )
        self.track = track
        return True


"""Utils"""


class FrictionDetector(Box2D.b2ContactListener):
    def __init__(self, env):
        Box2D.b2ContactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            # print tile.road_friction, "ADD", len(obj.tiles)
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)
            # print tile.road_friction, "DEL", len(obj.tiles) -- should delete to zero when on grass (this works)


def check_pyglet():
    from pyglet.window import key

    a = np.array([0.0, 0.0, 0.0])

    def key_press(k, _mod):
        if k == key.LEFT:
            a[0] = +1.0
        if k == key.RIGHT:
            a[0] = -1.0
        if k == key.UP:
            a[1] = +1.0
        if k == key.DOWN:
            a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

    def key_release(k, _mod):
        if k == key.LEFT and a[0] == +1.0:
            a[0] = 0
        if k == key.RIGHT and a[0] == -1.0:
            a[0] = 0
        if k == key.UP:
            a[1] = 0
        if k == key.DOWN:
            a[2] = 0

    env = CarRacingFix()
    env.render()
    env.viewer.window.on_key_press = key_press
    env.viewer.window.on_key_release = key_release
    record_video = False
    if record_video:
        from gym.wrappers.monitor import Monitor

        env = Monitor(env, "/tmp/video-test", force=True)
    if_open = True
    while if_open:
        env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            s, r, done, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or done:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
                # import matplotlib.pyplot as plt
                # plt.imshow(s)
                # plt.savefig("test.jpeg")
            steps += 1
            if_open = env.render()
            if done or not if_open:
                break
    env.close()


def check_env():
    # env = gym.make('CarRacing-v0')
    env = CarRacingFix()  # deepcopy(CarRacingFix()), raise gym.error.Error
    env.render()
    state = env.reset()
    print(state.shape)

    import cv2

    for i in range(256):
        action = (0.0, 1.0, 0.0) if i < 128 else (0.0, 0.0, 0.0)
        stack_state, reward, done, _ = env.step(action)
        cv2.imshow("", stack_state[::-1, :, 3:6])
        cv2.waitKey(1)
        env.render()

        if done:
            # cv2.waitKey(321)
            break


if __name__ == "__main__":
    # check_pyglet()
    check_env()
