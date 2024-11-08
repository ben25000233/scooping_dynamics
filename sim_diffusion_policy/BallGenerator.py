# -*- coding: utf-8 -*-
import xml.etree.ElementTree as gfg


class BallGenerator(object):

    def __init__(self):
        pass

    def generate(self, file_name=None, ball_radius=None, ball_mass=None, type = "solid"):

        root = gfg.Element("robot", name="ball")

        link = gfg.Element("link", name="ball")
        root.append(link)
        visual = gfg.Element("visual")
        link.append(visual)
        gfg.SubElement(visual, "origin", xyz="0 0 0")
        geometry = gfg.Element("geometry")
        visual.append(geometry)
        if type == "solid":
            gfg.SubElement(geometry, "sphere", radius=str(ball_radius))
        else : 
            gfg.SubElement(geometry, "sphere", radius=str(0))
        collision = gfg.Element("collision")
        link.append(collision)
        gfg.SubElement(collision, "origin", xyz="0 0 0")
        geometry = gfg.Element("geometry")
        collision.append(geometry)
        gfg.SubElement(geometry, "sphere", radius=str(ball_radius))
        inertial = gfg.Element("inertial")
        link.append(inertial)
        gfg.SubElement(inertial, "mass", value=str(ball_mass))
        gfg.SubElement(inertial, "inertia", ixx="0.00", ixy="0.0", ixz="0.0", iyy="0.00", iyz="0.0", izz="0.00")

        self.make_file(file_name, root)

    def make_file(self, file_name, root):
        tree = gfg.ElementTree(root)
        with open(f"urdf/{file_name}", "wb") as files:
            tree.write(files)


if __name__ == "__main__":
    urdfGenerator = BallGenerator()
    urdfGenerator.generate(file_name="BallHLS.urdf", ball_radius=0.01, ball_mass=0.001)
