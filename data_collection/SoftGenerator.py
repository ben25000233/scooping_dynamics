# -*- coding: utf-8 -*-
import xml.etree.ElementTree as gfg


class SoftGenerator(object):

    def __init__(self):
        pass

    def generate(self, file_name=None, scale="0.007", density="1e4", youngs = "1e10"):

        root = gfg.Element("robot", name="isosphere")

        link = gfg.Element("link", name="soft")
        root.append(link)
        fem = gfg.Element("fem")
        link.append(fem)
        gfg.SubElement(fem, "origin", rpy="0.0 0.0 0.0", xyz="0 0 0")
        gfg.SubElement(fem, "density", value=str(density))
        gfg.SubElement(fem, "youngs", value=str(youngs))
        gfg.SubElement(fem, "poissons", value="0")
        gfg.SubElement(fem, "damping", value="0")
        gfg.SubElement(fem, "tetmesh", filename="icosphere.tet")
        gfg.SubElement(fem, "scale", value=str(scale))

        self.make_file(file_name, root)

    def make_file(self, file_name, root):
        tree = gfg.ElementTree(root)
        with open(f"urdf/{file_name}", "wb") as files:
            tree.write(files)

