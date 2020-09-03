import copy
import math
# import sys

# import gym
import numpy as np
from gym import spaces
# from gym.utils import colorize, seeding
from gym.utils import seeding
# from six.moves import xrange
import Box2D
from Box2D.b2 import (circleShape, contactListener, edgeShape, fixtureDef, polygonShape,
                      revoluteJointDef)

MAX_AGENTS = 40

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

PACKAGE_POLY = [(-120, 5), (120, 5), (120, -5), (-120, -5)]

PACKAGE_LENGTH = 240

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5

WALKER_SEPERATION = 10  # in steps


class ContactDetector(contactListener):

    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        # if walkers fall on ground
        for i, walker in enumerate(self.env.walkers):
            if walker.hull == contact.fixtureA.body:
                if self.env.package != contact.fixtureB.body:
                    self.env.fallen_walkers[i] = True
            if walker.hull == contact.fixtureB.body:
                if self.env.package != contact.fixtureA.body:
                    self.env.fallen_walkers[i] = True

        # if package is on the ground
        if self.env.package == contact.fixtureA.body:
            if contact.fixtureB.body not in [w.hull for w in self.env.walkers]:
                self.env.game_over = True
        if self.env.package == contact.fixtureB.body:
            if contact.fixtureA.body not in [w.hull for w in self.env.walkers]:
                self.env.game_over = True

            #    self.env.game_over = True
        for walker in self.env.walkers:
            for leg in [walker.legs[1], walker.legs[3]]:
                if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                    leg.ground_contact = True

    def EndContact(self, contact):
        for walker in self.env.walkers:
            for leg in [walker.legs[1], walker.legs[3]]:
                if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                    leg.ground_contact = False


class Agent:

    def __new__(cls, *args, **kwargs):
        agent = super(Agent, cls).__new__(cls)
        return agent

    @property
    def observation_space(self):
        raise NotImplementedError()

    @property
    def action_space(self):
        raise NotImplementedError()

    def __str__(self):
        return '<{} instance>'.format(type(self).__name__)


class BipedalWalker(Agent):

    def __init__(self, world, init_x=TERRAIN_STEP * TERRAIN_STARTPAD / 2,
                 init_y=TERRAIN_HEIGHT + 2 * LEG_H, n_walkers=2, one_hot=False):
        self.world = world
        self._n_walkers = n_walkers
        self.one_hot = one_hot
        self.hull = None
        self.init_x = init_x
        self.init_y = init_y
        self._seed()

    def _destroy(self):
        if not self.hull:
            return
        self.world.DestroyBody(self.hull)
        self.hull = None
        for leg in self.legs:
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):
        self._destroy()

        init_x = self.init_x
        init_y = self.init_y
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x002,
                # maskBits=(0x001 & 0x002),  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.hull.color1 = (0.5, 0.4, 0.9)
        self.hull.color2 = (0.3, 0.3, 0.5)
        self.hull.ApplyForceToCenter((self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0),
                                     True)

        self.legs = []
        self.joints = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=fixtureDef(shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)), density=1.0,
                                    restitution=0.0, categoryBits=0x002,
                                    maskBits=0x001)  # collide with ground only
            )
            leg.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            leg.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(
                bodyA=self.hull,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=i,
                lowerAngle=-0.8,
                upperAngle=1.1, )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN), angle=(i * 0.05),
                fixtures=fixtureDef(shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
                                    density=1.0, restitution=0.0, categoryBits=0x0020,
                                    maskBits=0x001))
            lower.color1 = (0.6 - i / 10., 0.3 - i / 10., 0.5 - i / 10.)
            lower.color2 = (0.4 - i / 10., 0.2 - i / 10., 0.3 - i / 10.)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-1.6,
                upperAngle=-0.1, )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self.drawlist = self.legs + [self.hull]

        class LidarCallback(Box2D.b2.rayCastCallback):

            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return 1
                self.p2 = point
                self.fraction = fraction
                return 0

        self.lidar = [LidarCallback() for _ in range(10)]

    def apply_action(self, action):

        self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
        self.joints[0].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1))
        self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
        self.joints[1].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1))
        self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
        self.joints[2].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1))
        self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
        self.joints[3].maxMotorTorque = float(MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1))

    def get_observation(self):
        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE)
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            self.joints[0].
                angle,
            # This will give 1.1 on high up, but it's still OK (and there should be spikes on hiting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0
        ]

        state += [l.fraction for l in self.lidar]
        assert len(state) == 24

        return state

    @property
    def observation_space(self):
        # 24 original obs (joints, etc), 2 displacement obs for each neighboring walker, 3 for package, 1 ID
        idx = MAX_AGENTS if self.one_hot else 1  # TODO
        return spaces.Box(low=-np.inf, high=np.inf, shape=(24 + 4 + 3 + idx,))

    @property
    def action_space(self):
        return spaces.Box(low=-1, high=1, shape=(4,))


class MultiWalkerEnv:
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': FPS}

    hardcore = False

    def __init__(self, n_walkers=3, position_noise=1e-3, angle_noise=1e-3, reward_mech='local',
                 forward_reward=1.0, fall_reward=-100.0, drop_reward=-100.0, terminate_on_fall=True,
                 one_hot=False):
        # reward_mech is 'global' for cooperative game (same reward for every agent)

        self.n_walkers = n_walkers
        self.position_noise = position_noise
        self.angle_noise = angle_noise
        self._reward_mech = reward_mech
        self.forward_reward = forward_reward
        self.fall_reward = fall_reward
        self.drop_reward = drop_reward
        self.terminate_on_fall = terminate_on_fall
        self.one_hot = one_hot
        self.setup()

    def get_param_values(self):
        return self.__dict__

    def setup(self):
        self.seed()
        self.viewer = None

        self.world = Box2D.b2World()
        self.terrain = None

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.start_x = [
            init_x + WALKER_SEPERATION * i * TERRAIN_STEP for i in range(self.n_walkers)
        ]
        self.walkers = [
            BipedalWalker(self.world, init_x=sx, init_y=init_y, one_hot=self.one_hot)
            for sx in self.start_x
        ]
        self.num_agents = len(self.walkers)
        self.observation_space = [agent.observation_space for agent in self.walkers]
        self.action_space = [agent.action_space for agent in self.walkers]

        self.package_scale = self.n_walkers / 1.75
        self.package_length = PACKAGE_LENGTH / SCALE * self.package_scale

        self.total_agents = self.n_walkers

        self.prev_shaping = np.zeros(self.n_walkers)
        self.prev_package_shaping = 0.0

        self.terrain_length = int(TERRAIN_LENGTH * self.n_walkers * 1 / 8.)

        self.reset()

    @property
    def agents(self):
        return self.walkers

    @property
    def reward_mech(self):
        return self._reward_mech

    def seed(self, seed=None):
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def _destroy(self):
        if not self.terrain:
            return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        self.world.DestroyBody(self.package)
        self.package = None

        for walker in self.walkers:
            walker._destroy()

    def close(self):
        pass

    def reset(self):
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.fallen_walkers = np.zeros(self.n_walkers, dtype=np.bool)
        self.prev_shaping = np.zeros(self.n_walkers)
        self.prev_package_shaping = 0.0
        self.scroll = 0.0
        self.lidar_render = 0

        W = VIEWPORT_W / SCALE
        H = VIEWPORT_H / SCALE

        self._generate_package()
        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        self.drawlist = copy.copy(self.terrain)

        self.drawlist += [self.package]

        for walker in self.walkers:
            walker._reset()
            self.drawlist += walker.legs
            self.drawlist += [walker.hull]

        return self.step(np.array([0, 0, 0, 0] * self.n_walkers))[0]

    def step(self, actions):
        act_vec = np.reshape(actions, (self.n_walkers, 4))
        assert len(act_vec) == self.n_walkers
        for i in range(self.n_walkers):
            self.walkers[i].apply_action(act_vec[i])

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        obs = [walker.get_observation() for walker in self.walkers]

        xpos = np.zeros(self.n_walkers)
        obs = []
        done = False
        rewards = np.zeros(self.n_walkers)

        for i in range(self.n_walkers):
            pos = self.walkers[i].hull.position
            x, y = pos.x, pos.y
            xpos[i] = x

            wobs = self.walkers[i].get_observation()
            nobs = []
            for j in [i - 1, i + 1]:
                # if no neighbor (for edge walkers)
                if j < 0 or j == self.n_walkers:
                    nobs.append(0.0)
                    nobs.append(0.0)
                else:
                    xm = (self.walkers[j].hull.position.x - x) / self.package_length
                    ym = (self.walkers[j].hull.position.y - y) / self.package_length
                    nobs.append(np.random.normal(xm, self.position_noise))
                    nobs.append(np.random.normal(ym, self.position_noise))
            xd = (self.package.position.x - x) / self.package_length
            yd = (self.package.position.y - y) / self.package_length
            nobs.append(np.random.normal(xd, self.position_noise))
            nobs.append(np.random.normal(yd, self.position_noise))
            nobs.append(np.random.normal(self.package.angle, self.angle_noise))
            # ID
            if self.one_hot:
                nobs.extend(np.eye(MAX_AGENTS)[i])
            else:
                nobs.append(float(i) / self.n_walkers)
            obs.append(np.array(wobs + nobs))

            # shaping = 130 * pos[0] / SCALE
            shaping = 0.0
            shaping -= 5.0 * abs(wobs[0])
            rewards[i] = shaping - self.prev_shaping[i]
            self.prev_shaping[i] = shaping

        package_shaping = self.forward_reward * 130 * self.package.position.x / SCALE
        rewards += (package_shaping - self.prev_package_shaping)
        self.prev_package_shaping = package_shaping

        self.scroll = xpos.mean() - VIEWPORT_W / SCALE / 5 - (self.n_walkers - 1
                                                              ) * WALKER_SEPERATION * TERRAIN_STEP

        done = False
        if self.game_over or pos[0] < 0:
            rewards += self.drop_reward
            done = True
        if pos[0] > (self.terrain_length - TERRAIN_GRASS) * TERRAIN_STEP:
            done = True
        rewards += self.fall_reward * self.fallen_walkers
        if self.terminate_on_fall and np.sum(self.fallen_walkers) > 0:
            done = True

        if self.reward_mech == 'local':
            return obs, rewards, [done] * self.n_walkers, {}
        return obs, [rewards.mean()] * self.n_walkers, [done] * self.n_walkers, {}

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        render_scale = 0.75

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
        self.viewer.set_bounds(self.scroll,
                               VIEWPORT_W / SCALE * self.package_scale * render_scale + self.scroll,
                               0, VIEWPORT_H / SCALE * self.package_scale * render_scale)

        self.viewer.draw_polygon([
            (self.scroll, 0),
            (self.scroll + VIEWPORT_W * self.package_scale / SCALE * render_scale, 0),
            (self.scroll + VIEWPORT_W * self.package_scale / SCALE * render_scale,
             VIEWPORT_H / SCALE * self.package_scale * render_scale),
            (self.scroll, VIEWPORT_H / SCALE * self.package_scale * render_scale),
        ], color=(0.9, 0.9, 1.0))
        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2:
                continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE * self.package_scale:
                continue
            self.viewer.draw_polygon([(p[0] + self.scroll / 2, p[1]) for p in poly], color=(1, 1,
                                                                                            1))
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE * self.package_scale:
                continue
            self.viewer.draw_polygon(poly, color=color)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        for walker in self.walkers:
            if i < 2 * len(walker.lidar):
                l = walker.lidar[i] if i < len(walker.lidar) else walker.lidar[len(walker.lidar)
                                                                               - i - 1]
                self.viewer.draw_polyline([l.p1, l.p2], color=(1, 0, 0), linewidth=1)

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 30, color=obj.color2, filled=False,
                                            linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        flagy1 = TERRAIN_HEIGHT
        flagy2 = flagy1 + 50 / SCALE
        x = TERRAIN_STEP * 3
        self.viewer.draw_polyline([(x, flagy1), (x, flagy2)], color=(0, 0, 0), linewidth=2)
        f = [(x, flagy2), (x, flagy2 - 10 / SCALE), (x + 25 / SCALE, flagy2 - 5 / SCALE)]
        self.viewer.draw_polygon(f, color=(0.9, 0.2, 0))
        self.viewer.draw_polyline(f + [f[0]], color=(0, 0, 0), linewidth=2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def _generate_package(self):
        init_x = np.mean(self.start_x)
        init_y = TERRAIN_HEIGHT + 3 * LEG_H
        self.package = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x * self.package_scale / SCALE, y / SCALE)
                                             for x, y in PACKAGE_POLY]),
                density=1.0,
                friction=0.5,
                categoryBits=0x004,
                # maskBits=0x001,  # collide only with ground
                restitution=0.0)  # 0.99 bouncy
        )
        self.package.color1 = (0.5, 0.4, 0.9)
        self.package.color2 = (0.3, 0.3, 0.5)

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state = GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []
        for i in range(self.terrain_length):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD:
                    velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                y += velocity

            elif state == PIT and oneshot:
                counter = self.np_random.randint(3, 5)
                poly = [
                    (x, y),
                    (x + TERRAIN_STEP, y),
                    (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                    (x, y - 4 * TERRAIN_STEP),
                ]
                t = self.world.CreateStaticBody(fixtures=fixtureDef(
                    shape=polygonShape(vertices=poly), friction=FRICTION))
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)
                t = self.world.CreateStaticBody(fixtures=fixtureDef(shape=polygonShape(
                    vertices=[(p[0] + TERRAIN_STEP * counter, p[1]) for p in poly]),
                    friction=FRICTION))
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * TERRAIN_STEP

            elif state == STUMP and oneshot:
                counter = self.np_random.randint(1, 3)
                poly = [
                    (x, y),
                    (x + counter * TERRAIN_STEP, y),
                    (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP),
                    (x, y + counter * TERRAIN_STEP),
                ]
                t = self.world.CreateStaticBody(fixtures=fixtureDef(
                    shape=polygonShape(vertices=poly), friction=FRICTION))
                t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                self.terrain.append(t)

            elif state == STAIRS and oneshot:
                stair_height = +1 if self.np_random.rand() > 0.5 else -1
                stair_width = self.np_random.randint(4, 5)
                stair_steps = self.np_random.randint(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (x + (s * stair_width) * TERRAIN_STEP,
                         y + (s * stair_height) * TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * TERRAIN_STEP,
                         y + (s * stair_height) * TERRAIN_STEP),
                        (x + ((1 + s) * stair_width) * TERRAIN_STEP,
                         y + (-1 + s * stair_height) * TERRAIN_STEP),
                        (x + (s * stair_width) * TERRAIN_STEP,
                         y + (-1 + s * stair_height) * TERRAIN_STEP),
                    ]
                    t = self.world.CreateStaticBody(fixtures=fixtureDef(
                        shape=polygonShape(vertices=poly), friction=FRICTION))
                    t.color1, t.color2 = (1, 1, 1), (0.6, 0.6, 0.6)
                    self.terrain.append(t)
                counter = stair_steps * stair_width

            elif state == STAIRS and not oneshot:
                s = stair_steps * stair_width - counter - stair_height
                n = s / stair_width
                y = original_y + (n * stair_height) * TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.randint(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if state == GRASS and hardcore:
                    state = self.np_random.randint(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(self.terrain_length - 1):
            poly = [(self.terrain_x[i], self.terrain_y[i]),
                    (self.terrain_x[i + 1], self.terrain_y[i + 1])]
            t = self.world.CreateStaticBody(fixtures=fixtureDef(
                shape=edgeShape(vertices=poly),
                friction=FRICTION,
                categoryBits=0x0001, ))
            color = (0.3, 1.0 if i % 2 == 0 else 0.8, 0.3)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (0.4, 0.6, 0.3)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly = []
        for i in range(self.terrain_length // 20):
            x = self.np_random.uniform(0, self.terrain_length) * TERRAIN_STEP
            y = VIEWPORT_H / SCALE * 3 / 4
            poly = [(x + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5) + self.np_random.uniform(
                0, 5 * TERRAIN_STEP), y + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5) +
                     self.np_random.uniform(0, 5 * TERRAIN_STEP)) for a in range(5)]
            x1 = min([p[0] for p in poly])
            x2 = max([p[0] for p in poly])
            self.cloud_poly.append((poly, x1, x2))


class GymMultiWalkerEnv:
    metadata = {'render.modes': ['human']}

    def __init__(self, *args, **kwargs):
        super(GymMultiWalkerEnv, self).__init__()
        self.env = MultiWalkerEnv(*args, **kwargs)

        self.num_agents = self.env.num_agents
        self.agent_ids = list(range(self.num_agents))
        # spaces
        self.action_space_dict = dict(zip(self.agent_ids, self.env.action_space))
        self.observation_space_dict = dict(zip(self.agent_ids, self.env.observation_space))
        self.steps = 0
        self.reset()

    def convert_to_dict(self, list_of_list):
        return dict(zip(self.agent_ids, list_of_list))

    def reset(self):
        observation = self.env.reset()
        self.steps = 0
        return self.convert_to_dict(observation)

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()

    def step(self, action_dict):
        # unpack actions
        actions = [0.0 for _ in range(len(action_dict))]

        for agent_id in self.agent_ids:
            if any(np.isnan(action_dict[agent_id])):
                action_dict[agent_id] = [0 for _ in action_dict[agent_id]]
            elif not self.action_space_dict[agent_id].contains(action_dict[agent_id]):
                raise Exception('Action for agent {} must be in {}. \
                                It is currently {}'.format(agent_id, self.action_space_dict[agent_id],
                                                           action_dict[agent_id]))
            actions[agent_id] = action_dict[agent_id]

        observation, reward, done, info = self.env.step(actions)

        if self.steps >= 500:
            done = [True] * self.num_agents

        observation_dict = self.convert_to_dict(observation)
        reward_dict = self.convert_to_dict(reward)
        info_dict = self.convert_to_dict(info)
        done_dict = self.convert_to_dict(done)
        done_dict["__all__"] = done[0]

        self.steps += 1

        return observation_dict, reward_dict, done_dict, info_dict


def modify_multi_to_single_walker_env(env):
    def decorator_step(env_step):
        def new_env_step(action):
            action = action.clip(-1, 1)  # fix bug in class GymMultiWalkerEnv
            action_dict = {0: action[0:4],
                           1: action[4:8],
                           2: action[8:12]}
            state_dict, reward_dict, done_dict, info = env_step(action_dict)
            state = np.hstack(tuple(state_dict.values()))
            reward = sum(reward_dict.values())
            done = any(done_dict.values())
            return state, reward, done, info

        return new_env_step

    env.step = decorator_step(env.step)

    def decorator_reset(env_reset):
        def new_env_reset():
            state_dict = env_reset()
            state = np.hstack(tuple(state_dict.values()))
            return state

        return new_env_reset

    env.reset = decorator_reset(env.reset)
    return env


"""run"""


def run_demo__multi_walker():
    # from sisl_games.multiwalker.multiwalker import env as custom_env

    """sisl's multi-walker"""
    env = GymMultiWalkerEnv()
    state_l = [i.shape[0] for i in env.observation_space_dict.values()]
    action_l = [i.shape[0] for i in env.action_space_dict.values()]
    print(state_l)
    print(action_l)

    state = env.reset()

    max_step = 1000
    total_reward = 0
    for i in range(max_step):
        action_dict = dict([(i, np.random.uniform(-1.0, +1.0, n)) for i, n in enumerate(action_l)])
        observations, rewards, dones, info = env.step(action_dict)
        env.render()
        total_reward += sum(rewards.values())
        done = any(list(dones.values()))
        # if sum(rewards.values()) > 0:
        #     print("rewards", rewards)
        rs = list(rewards.values())
        print("R:", sum(rs), np.array(rs).round(2))
        if done:
            break

    print("done", total_reward)
    # env.close()

    """multi-walker modify by YonV1943"""  # wait to check
    # env = GymMultiWalkerEnv()
    env = modify_multi_to_single_walker_env(env)

    state_l = [i.shape[0] for i in env.observation_space_dict.values()]
    action_l = [i.shape[0] for i in env.action_space_dict.values()]
    print(state_l)
    print(action_l)

    state = env.reset()

    max_step = 1000
    total_reward = 0
    import time
    t0 = time.time()
    for i in range(max_step):
        action = np.random.uniform(-1, 1, size=4 * 3)
        state, reward, done, info = env.step(action)
        # env.render()
        total_reward += reward
        print(f"R {reward:.3f}")
        if done:
            break

    print(time.time() - t0)
    print("done", total_reward)
    env.close()


def run_continuous_action(gpu_id=None):  # fail
    # import AgentZoo as Zoo
    from AgentRun import *
    import beta1 as Zoo  # params-share

    """online policy"""  # plan to check args.max_total_step
    args = Arguments(rl_agent=Zoo.AgentInterGAE, gpu_id=gpu_id)
    args.net_dim = 2 ** 8
    args.max_memo = 2 ** 13  # 12
    args.batch_size = 2 ** 9
    args.repeat_times = 2 ** 5  # 4

    args.env_name = 'MultiWalker'
    args.max_total_step = int(4e6 * 4)
    args.init_for_training()
    train_agent(**vars(args))
    exit()


if __name__ == '__main__':
    run_demo__multi_walker()
    # run_continuous_action()
    # from multiwalker_base import run_demo__multi_walker
    # run_demo__multi_walker()
