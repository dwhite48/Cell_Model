################################################################################
# Name:   Analysis Functions
# Author: Douglas E. White
# Date:   11/19/2013
################################################################################

# Has functions for analyzing network time series
import simulationMath as simMath
from visual import *
import time as t_m
from PIL import ImageGrab, Image, ImageDraw, ImageFont
import visvis as vv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Colormaps import colormap
import subprocess

def compute_clusters(network, comparator, lower_lim = 1):
    """ Evaluates clusters which may be present based on the comparison
        function provided. Returns a list of these networks as networkX
        graphs.
    """
    networks = []
    #first get all the nodes in the network
    nodes = network.nodes()
    #copy the network
    g = network.copy()
    nodes = g.nodes()
    print("Examining " + repr(len(nodes)) + " nodes")
    for i in range(0, len(nodes)):
        #tag all the nodes with the examined tag
        nodes[i].examined = False
        nodes[i].in_list = False
    #save the networks
    clusters = []
    #keep track fo the nodes to add
    nodes_to_check = []
    #now iterate
    for i in range(0, len(nodes)):
        #if the node has not been examined
        if(not nodes[i].examined):
            #add the first node to the node to check
            nodes_to_check = [nodes[i]]
            #now iterate over the list
            j = 0
            while(j < len(nodes_to_check)):
                #check to see if this is examined already
                if(not nodes_to_check[j].examined):
                    #get the neighbors and add these to the list
                    nbs = g.neighbors(nodes_to_check[j])
                    nodes_to_check[j].examined = True
                    #now loop over all fo the neighbros
                    for k in range(0, len(nbs)):
                        #check to see it is in the examine list
                        if(not nbs[k].in_list):
                            #check to see if it fits the criterion
                            if(comparator(nbs[k])):
                                #then add it to the list
                                nodes_to_check.append(nbs[k])
                                nbs[k].in_list = True
                #increment j
                j += 1
            #now we have all of the nodes in the nodes to check list
            #make a new graph
            if(len(nodes_to_check) > lower_lim):
                n = nx.Graph()
                #add these nodes to the graph
                n.add_nodes_from(nodes_to_check)
                #also add all the connections
                for i in range(0, len(nodes_to_check)):
                    #get the neighbors of n
                    node = nodes_to_check[i]
                    nbs = network.neighbors(node)
                    #add them all to the network
                    for other in nbs:
                        n.add_edge(node, other)
                #add this to the list
                clusters.append(n)
    #now return the clusters
    return clusters

def draw_color_bar(mn, mx, width, height, font):
    """ Draws a color bar from mn to max using the specified font. The width
        and height parameters govern the shape of the color bar, NOT the total
        size of the returned image
    """
    half = '%.4f' % (mx/2.)
    mn = '%.4f' % mn
    mx = '%.4f' % mx
    #make a dummy image form drawing and sizing
    im = Image.new("RGB", (0, 0), "white")
    draw = ImageDraw.Draw(im)
    #figure out the sizing of the color bar
    tw, th = draw.textsize(half, font = font)
    w = width + tw + 10
    h = height + th + 10 #same height as the heat map, plus the height of the text
    #new image and drawing object
    color_bar = Image.new("RGB", (w,h), "white")
    draw = ImageDraw.Draw(color_bar)
    #draw the scaled color bar
    for i in range(0, height):
        #draw from top to bottom the color gradient used
        R = min((256./(height/3.))*(height - i), 256)
        G = min(max(((256./(height/3.))*(height - i - height/3.)), 0), 256)
        B = min(max(((256./(height/3.))*(height - i - 2.*height/3.)), 0), 256)
        col = (int(R), int(G), int(B))
        pos = (0, i + int(th/2) + 5, w - tw - 10, i + int(th/2) + 5)
        draw.line(pos, fill = col)
    #draw the text for the scale on this
    #max
    pos = (width + 5, int(th/2))
    draw.text(pos, mx, font = font, fill = (0,0,0))
    #middle
    pos = (width + 5, int(h/2) - int(th/2)) 
    draw.text(pos, half, font = font, fill = (0,0,0))
    #min
    pos = (width + 5, height + int(th/2) - 5) 
    draw.text(pos, mn, font = font, fill = (0,0,0))
    return color_bar
    
def ZeroPad(maxTime, time):
    """ Zero pads the current number
        Returns the current number as an string weith the correct number
        of 0s appended
    """
    #figure out the max number of diogits in the final time
    l = math.log(maxTime, 10)
    if(abs(l - math.ceil(l)) < 0.000001):
        l = math.ceil(l)
    d1 = int(math.floor(l))
    #figure out the number of diogits in the current time
    if(time == 0):
        d2 = 0
    else:
        l = math.log(time, 10)
        if(abs(l - math.ceil(l)) < 0.000001):
            l = math.ceil(l)
        d2 = int(math.floor(l))
    #now append 0's the fron of the current time proportional to the difference
    s = ""
    for i in range(0, d1-d2):
        s = s + "0"
    s = s + repr(time)
    return s

def get_boundary_nodes(network, subnetwork):
    #gets the list of nodes which sit on the edge of the network
    nx.node_boundary(network, subnetwork)

def GetCenter(network):
    #gets the center of a list of agents in a network
    agents = network.nodes()
    cent = [0,0,0]
    for i in range(0, len(agents)):
        cent = simMath.AddVec(cent, agents[i].location)
    #Normalize
    cent = simMath.ScaleVec(cent, 1.0/len(agents))
    #now return
    return cent

def getAverageRadialDistance(network, center):
    agents = network.nodes()
    cent = [0,0,0]
    radii = []
    for i in range(0, len(agents)):
        radius = simMath.SubtractVec(agents[i].location, cent)
        #add tot he list
        radii.append(simMath.Mag(radius))
    #now return the average and the stdev
    return np.average(radii), np.std(radii)    

def GetRadiusandCenter(network):
    #first get the center
    center = GetCenter(network)
    #gets the maximum radial size of the network
    agents = network.nodes()
    radius = -1
    for i in range(0, len(agents)):
        r = simMath.SubtractVec(center, agents[i].location)
        radius = max(simMath.Mag(r), radius)
    #now return
    return radius, center

def draw_cells_2D(network, thickness = 10, plane = 0):
    """ Filter a 3D cell set down to a 2D cell plane of thickness
        thickness*2. The plane to used is specified by the plane
        function as follows:
        0 - xy
        1 - xz
        2 - yz
        Returns - a network with the contained nodes
    """
    if(plane == 0):
        #takes a networkX object to use to draw with
        scene.center = GetCenter(network)
        #for each agent in the nodes draw a sphere
        agents = network.nodes()
        for agent in agents:
            #calculate the distance in z
            if(abs(agent.location[2] - scene.center[2]) < thickness):
                #then you can plot it
                if(agent.sim_type == "stemcell"):
                    if(agent.state == "U" or agent.state == "T"):
        ##                col = [0, 1.0, 1.0]
        ##                op = 0.01
                        col = color.cyan
                        op = 1.0
        ##            if(agent.state == "T"):
        ##                col = color.green
        ##                op = 1.0
                    if(agent.state == "D"):
                        col = color.blue
                        op = 1.0
                    if(agent.state == "A"):
                        col = color.red
                        op = 1.0
                    if("N+" in agent.state):
        ##                col = color.yellow
                        col = color.blue
                        op = 1.0
                    if(agent.state == "O+N-"):
        ##                col = color.red
                        col = color.blue
                        op = 1.0
                    if(agent.state == "O-N-"):
        ##                col = color.blue
                        col = color.blue
                    op = 1.0
##                    ratio = agent.sol_count / agent.sol_count_max
##                    col = (0,1 - ratio,1)
                elif(agent.sim_type == "stemcellmes"):
                    if(agent.state == "U" or agent.state == "T"):
                        col = [0, 1.0, 1.0]
                        op = 0.1
                    if(agent.state == "T"):
                        col = color.green
                        op = 1.0
                    if(agent.state == "D"):
                        col = color.blue
                        op = 0.5
                    if(agent.state == "A"):
                        col = color.red
                        op = 1.0
                elif(agent.sim_type == "NueronalStemCell"):
                    if(agent.state == "U" or agent.state == "T1"):
                        col = color.blue
                        op = 0.1
                    if(agent.state == "MN"):
                        col = color.green
                        op = 1.0
                    if(agent.state == "NSC"):
                        col = color.red
                        op = 1.0
                    if(agent.state == "G"):
                        col = color.cyan
                        op = 1.0
                    if(agent.state == "T2"):
##                        col = color.yellow
                        col = color.green
##                        col = color.red
                        op = 1.0
                    if(agent.state == "T3"):
                        col = [.5, 0, .5]
                        op = 1.0
                elif(agent.sim_type == "dividingcell"):
                    col = color.cyan
                    op = 1.0
                elif(agent.sim_type == "fishcell"):
                    if(agent.state == "n"):
                        col = color.blue
                        op = 0.1
                    if(agent.state == "ne"):
                        col = color.green
                        op = 1.0
                    if(agent.state == "dlx3b"):
                        col = color.red
                        op = 0.5
                    if(agent.state == "np"):
                        col = color.yellow
                        op = 1.0    
                elif(agent.sim_type == "microparticle"):
                    col = color.yellow
                    op = 1.0
                elif(agent.sim_type == "nueron"):
                    col = color.green
                    op = 1.0
                elif(agent.sim_type == "nuerite"):
                    col = color.green
                    op = 1.0
                    #also draw the cons
                    p1 = agent.parent.location
                    p2 = agent.location
                    axis = simMath.SubtractVec(p1, p2)
                    #draw the cons
                    cylinder(pos = p2,
                         axis = axis,
                         radius = agent.radius,
                         color = col)
                #make a new sphere and add it to the dicitionary
                sphere(pos = agent.location,
                       radius = agent.radius,
                       color = col,
                       opacity = op)
        
    elif(plane == 1):
        pass
    elif(plane == 3):
        pass
    else:
        return None

def draw_cells(network):
    #takes a networkX object to use to draw with
    scene.center = GetCenter(network)
    print(scene.center)
    #for each agent in the nodes draw a sphere
    agents = network.nodes()
    print("Agents to draw: " + repr(len(agents)))
    for i in range(0, len(agents)):
        agent = agents[i]
        #Now color code based on the type
        if(agent.sim_type == "stemcell"):
            if(agent.state == "U" or agent.state == "T"):
##                col = [0, 1.0, 1.0]
##                op = 0.01
                col = color.blue
                op = 1.0
##            if(agent.state == "T"):
##                col = color.green
##                op = 1.0
            if(agent.state == "D"):
                col = color.blue
                op = 1.0
            if(agent.state == "A"):
                col = color.red
                op = 1.0
            if("N+" in agebt.state):
##                col = color.yellow
                col = color.blue
                op = 1.0
            if(agent.state == "O+N-"):
##                col = color.red
                col = color.blue
                op = 1.0
            if(agent.state == "O-N-"):
##                col = color.blue
                col = color.blue
                op = 1.0
        elif(agent.sim_type == "stemcellmes"):
            if(agent.state == "U" or agent.state == "T"):
                col = [0, 1.0, 1.0]
                op = 0.1
            if(agent.state == "T"):
                col = color.green
                op = 1.0
            if(agent.state == "D"):
                col = color.blue
                op = 0.5
            if(agent.state == "A"):
                col = color.red
                op = 1.0
        elif(agent.sim_type == "NueronalStemCell"):
            if(agent.state == "U" or agent.state == "T1"):
                col = color.blue
                op = 0.1
            if(agent.state == "MN"):
                col = color.green
                op = 1.0
            if(agent.state == "NSC"):
                col = color.red
                op = 0.5
            if(agent.state == "G"):
                col = color.cyan
                op = 1.0
            if(agent.state == "T2"):
                col = color.yellow
                op = 0.75
            if(agent.state == "T3"):
                col = [.5, 0, .5]
                op = .75
        elif(agent.sim_type == "fishcell"):
            if(agent.state == "n"):
                col = color.blue
                op = 0.5
            if(agent.state == "g"):
                col = color.green
                op = 1.0
            if(agent.state == "y"):
                col = color.yellow
                op = 0.5
            if(agent.state == "r"):
                col = color.red
                op = 1.0    
        elif(agent.sim_type == "dividingcell"):
            if(agent.state == "U" or agent.state == "T"):
                col = [0, 1.0, 1.0]
                op = 0.1
            if(agent.state == "T"):
                col = color.green
                op = 1.0
            if(agent.state == "D"):
                col = color.blue
                op = 0.5
            if(agent.state == "A"):
                col = color.red
                op = 1.0             
        elif(agent.sim_type == "microparticle"):
            col = color.yellow
            op = 1.0
        elif(agent.sim_type == "nueron"):
            col = color.green
            op = 1.0
        elif(agent.sim_type == "nuerite"):
            col = color.green
            op = 1.0
            #also draw the cons
            p1 = agent.parent.location
            p2 = agent.location
            axis = simMath.SubtractVec(p1, p2)
            #draw the cons
            cylinder(pos = p2,
                 axis = axis,
                 radius = agent.radius,
                 color = col)
        #make a new sphere and add it to the dicitionary
        sphere(pos = agent.location,
               radius = agent.radius,
               color = col,
               opacity = op)

def draw_cells_by_gradient(network, grad, color_map):
    #takes a networkX object to use to draw with
    scene.center = GetCenter(network)
    #for each agent in the nodes draw a sphere
    agents = network.nodes()
    print("Agents to draw: " + repr(len(agents)))
    for i in range(0, len(agents)):
        agent = agents[i]
        #get the value based on the objects
        val = agents[i].get_gradient_value(grad.name)
        if(val != -1):
            #get the color based on the color map values
            col = color_map.getColor(val) #get the RGBA
            #ok now get the gradient value at this point
            sphere(pos = agent.location,
                   radius = agent.radius,
                   color = (col[0], col[1], col[2]),
                   opacity = col[3])

def save_cell_images_by_gradient(save_path, data, gradient_name,
                                 rng = 250, scale_bar = 50):
    #deinfe the bounding box to take images at
    bbox = [10,30,590,570]
    #save all of the images
    images = []
    #gte the itmes
    times = data.get_times()
    #load a new font
    my_font = ImageFont.truetype("arial.ttf", 14)
    #first loop through to get the min and max of the gradient
    mn = 1000000000000000
    mx = -1
    for i in range(0, len(times)):
        grad = data.get_gradient_at_time(times[i], gradient_name)
        #now update mn, and mx
        mn = min(mn, np.min(grad.C))
        mx = max(mx, np.max(grad.C))
    print(mn, mx)
        
    #prelabel all of the nodes with a pram binding them to a sphere
    #now for each time get the network
    for i in range(0, len(times)):
        #make the drawing scene
        scene = display(x = 0, y = 0, width = 600, height = 600, autoscale = 0)
        scene.background = color.white
        scene.exit = False
        #also set the range
        scene.range = rng
        time = times[i]
        print(time)
        print("Loading...")
        #get the time point
        tp = data.get_time_point(time)
        grad = data.get_gradient_at_time(time, gradient_name)
        #set the colormap
        vals = {0:[0,0,0,.1], 1E-5:[1,0,0,1]}
        color_map = colormap(vals)
        #Now render the cells
        print("Drawing...")
        draw_cells_by_gradient(tp, grad, color_map)
        #wait one seconds
        t_m.sleep(1)
        print("Save the image")
        im = ImageGrab.grab(bbox)
        #write the time in the upper left hand corner
        draw = ImageDraw.Draw(im)
        draw.text((10,10),
                  "Time : " + repr(time),
                  font = my_font,
                  fill = (0,0,0))
        #save a scale bar in the upper right hand corner
        #figure out the wdith
        width = (600 / (rng*2.))*scale_bar
        #set the text to draw
        text = repr(scale_bar) + " um"
        #get the size for drawing
        w, h = draw.textsize(text, font = my_font)
        width = max(width, w)
        #make the text
        draw.text((580 - 10 - width, 10),
                  text,
                  font = my_font,
                  fill = (0,0,0))
        #now draw the line at 10 + 1.1*h
        draw.line(((580 - 10 - width, 10 + 1.75*h),
                   (580 - 10, 10 + 1.75*h)),
                  fill = (0,0,0))
        #get the zero padded image save name to order the images correctly
        s = ZeroPad(times[-1], time)
        im.save(save_path + "agents_" + gradient_name + "_" + s + ".png", "PNG")

        #delete the scene
        scene.visible = False
        del scene
    #finally save all the videos
    #get the number of digits to use
    s = s.split('.')
    s = len(s[0])
    subprocess.call(["C:\\users\\doug\\desktop\\CM3Dv4\\MakeMovie.bat",
                     save_path +"agents_" + gradient_name+ "_%0" + repr(s) + "d.0.png",
                     save_path +"agents_" + gradient_name + "_"])

def draw_cells_by_gradient_2D(network, grad, color_map, thickness = 10):
    #takes a networkX object to use to draw with
    scene.center = GetCenter(network)
    #for each agent in the nodes draw a sphere
    agents = network.nodes()
    print("Agents to draw: " + repr(len(agents)))
    for i in range(0, len(agents)):
        agent = agents[i]
        #get the value based on the objects
        if(abs(agent.location[2] - scene.center[2]) < thickness):
            val = agents[i].get_gradient_value(grad.name)
            if(val != -1):
                #get the color based on the color map values
                col = color_map.getColor(val) #get the RGBA
                #ok now get the gradient value at this point
                sphere(pos = agent.location,
                       radius = agent.radius,
                       color = (col[0], col[1], col[2]),
                       opacity = col[3])

def save_2D_cell_images_by_gradient(save_path, data, gradient_name,
                                     rng = 250, scale_bar = 50):
    #deinfe the bounding box to take images at
    bbox = [10,30,590,570]
    #save all of the images
    images = []
    #gte the itmes
    times = data.get_times()
    #load a new font
    my_font = ImageFont.truetype("arial.ttf", 14)
    #loop over these
    for i in range(0, len(times)):
        #make the drawing scene
        scene = display(x = 0, y = 0, width = 600, height = 600, autoscale = 0)
        scene.background = color.white
        scene.exit = False
        #also set the range
        scene.range = rng
        time = times[i]
        print(time)
        print("Loading...")
        #get the time point
        tp = data.get_time_point(time)
        #Get the gradient data
        grad = data.get_gradient_at_time(time, gradient_name)
        #set the colormap
        vals = {0:[0,0,1,1], 1E-5:[1,0,0,1]}
        color_map = colormap(vals)
        #Now render the cells
        print("Drawing...")
        draw_cells_by_gradient_2D(tp, grad, color_map)
        #wait one seconds
        t_m.sleep(1)
        print("Save the image")
        im = ImageGrab.grab(bbox)
        #write the time in the upper left hand corner
        draw = ImageDraw.Draw(im)
        draw.text((10,10),
                  "Time : " + repr(time),
                  font = my_font,
                  fill = (0,0,0))
        #save a scale bar in the upper right hand corner
        #figure out the wdith
        width = (600 / (rng*2.))*scale_bar
        #set the text to draw
        text = repr(scale_bar) + " um"
        #get the size for drawing
        w, h = draw.textsize(text, font = my_font)
        width = max(width, w)
        #make the text
        draw.text((580 - 10 - width, 10),
                  text,
                  font = my_font,
                  fill = (0,0,0))
        #now draw the line at 10 + 1.1*h
        draw.line(((580 - 10 - width, 10 + 1.75*h),
                   (580 - 10, 10 + 1.75*h)),
                  fill = (0,0,0))
        #get the zero padded image save name to order the images correctly
        s = ZeroPad(times[-1], time)
        im.save(save_path + "agents_2D_" + gradient_name + "_" + s +".png", "PNG")

        #delete the scene
        scene.visible = False
        del scene
    #now make the movie
    #get the number of digits to use
    s = s.split('.')
    s = len(s[0])
    subprocess.call(["C:\\users\\doug\\desktop\\CM3Dv4\\MakeMovie.bat",
             save_path + "agents_2D_" + gradient_name + "_" + "%0" + repr(s) + "d.0.png",
             save_path +"agents_2D_"])

def save_2D_cell_images(save_path, data, rng = 250, scale_bar = 50):
    #deinfe the bounding box to take images at
    bbox = [10,30,590,570]
    #save all of the images
    images = []
    #gte the itmes
    times = data.get_times()
    #load a new font
    my_font = ImageFont.truetype("arial.ttf", 14)
    #loop over these
    for i in range(0, len(times)):
        #make the drawing scene
        scene = display(x = 0, y = 0, width = 600, height = 600, autoscale = 0)
        scene.background = color.white
        scene.exit = False
        #also set the range
        scene.range = rng
        time = times[i]
        print(time)
        print("Loading...")
        #get the time point
        tp = data.get_time_point(time)
        #Now filter the time point in 2D
        
        #Now render the cells
        print("Drawing...")
        draw_cells_2D(tp)
        #wait one seconds
        t_m.sleep(1)
        print("Save the image")
        im = ImageGrab.grab(bbox)
        #write the time in the upper left hand corner
        draw = ImageDraw.Draw(im)
        draw.text((10,10),
                  "Time : " + repr(time),
                  font = my_font,
                  fill = (0,0,0))
        #save a scale bar in the upper right hand corner
        #figure out the wdith
        width = (600 / (rng*2.))*scale_bar
        #set the text to draw
        text = repr(scale_bar) + " um"
        #get the size for drawing
        w, h = draw.textsize(text, font = my_font)
        width = max(width, w)
        #make the text
        draw.text((580 - 10 - width, 10),
                  text,
                  font = my_font,
                  fill = (0,0,0))
        #now draw the line at 10 + 1.1*h
        draw.line(((580 - 10 - width, 10 + 1.75*h),
                   (580 - 10, 10 + 1.75*h)),
                  fill = (0,0,0))
        #get the zero padded image save name to order the images correctly
        s = ZeroPad(times[-1], time)
        im.save(save_path + "agents_2D_" + s +".png", "PNG")

        #delete the scene
        scene.visible = False
        del scene
    #now make the movie
    #get the number of digits to use
    s = s.split('.')
    s = len(s[0])
    subprocess.call(["C:\\users\\doug\\desktop\\CM3Dv4\\MakeMovie.bat",
             save_path +"agents_2D_" "%0" + repr(s) + "d.0.png",
             save_path +"agents_2D_"])
    

def save_cell_images(save_path, data, rng = 250, scale_bar = 50):
    #deinfe the bounding box to take images at
    bbox = [10,30,590,570]
    #save all of the images
    images = []
    #gte the itmes
    times = data.get_times()
    #load a new font
    my_font = ImageFont.truetype("arial.ttf", 14)
    #prelabel all of the nodes with a pram binding them to a sphere
    #now for each time get the network
    for i in range(0, len(times)):
        #make the drawing scene
        scene = display(x = 0, y = 0, width = 600, height = 600, autoscale = 0)
        scene.background = color.white
        scene.exit = False
        #get the time
        time = times[i]
        print(time)
        print("Loading...")
        #get the time point
        tp = data.get_time_point(time)
        #Now render the cells
        print("Drawing...")
        draw_cells(tp)
        #also set the range
        scene.range = rng
        #wait one seconds
        t_m.sleep(1)
        print("Save the image")
        im = ImageGrab.grab(bbox)
        #write the time in the upper left hand corner
        draw = ImageDraw.Draw(im)
        draw.text((10,10),
                  "Time : " + repr(time),
                  font = my_font,
                  fill = (0,0,0))
        #save a scale bar in the upper right hand corner
        #figure out the wdith
        width = (600 / (rng*2.))*scale_bar
        #set the text to draw
        text = repr(scale_bar) + " um"
        #get the size for drawing
        w, h = draw.textsize(text, font = my_font)
        width = max(width, w)
        #make the text
        draw.text((580 - 10 - width, 10),
                  text,
                  font = my_font,
                  fill = (0,0,0))
        #now draw the line at 10 + 1.1*h
        draw.line(((580 - 10 - width, 10 + 1.75*h),
                   (580 - 10, 10 + 1.75*h)),
                  fill = (0,0,0))
        #get the zero padded image save name to order the images correctly
        s = ZeroPad(times[-1], time)
        im.save(save_path + "agents_" + s +".png", "PNG")

        #delete the scene
        scene.visible = False
        del scene
    #now make the movie
    #get the number of digits to use
    s = s.split('.')
    s = len(s[0])
    subprocess.call(["C:\\users\\doug\\desktop\\CM3Dv4\\MakeMovie.bat",
             save_path +"agents_" "%0" + repr(s) + "d.0.png",
             save_path +"agents_"])

def save_2D_gradient_images(name, save_path, data,
                            axis = "xy"):
    """ Will save a 2D slice of the gradient specified by name along the given
        axis which by default is xy
    """
    #load the data
    times = data.get_times()
    #get the min and max overall
    mx = -1
    mn = 1000000000000
    #save the planes
    planes = []
    for i in range(0, len(times)):
        #get the gradient
        g = data.get_gradient_at_time(times[i], name)
        #now slice it depending on the axis
        plane = None
        #compute half plane vbalue for x and y
        x, y, z = g.C.shape
        if(axis == "xy"):
            plane = g.C[:, :, z/2]
        if(axis == "xz"):
            plane = g.C[:, y/2, :]
        if(axis == "yz"):
            plane = g.C[x/2 , :, :]
        #save the slice
        planes.append(plane)
        #compute the min and max for graphing
        mn = min(np.min(plane), mn)
        mx = max(np.max(plane), mx)
##        mx = 0.005
    #now make and save a figure for each slice
    for i in range(0, len(planes)):
        #set the min and max
        #draw the figure
        plt.imshow(planes[i],
                   vmin = mn,
                   vmax = mx,
                   interpolation = 'bicubic')
        #add a color bar
        plt.colorbar()
        #add a title
        plt.title(g.name + "   Time: " + repr(times[i]))
        #zeropad it
        s = ZeroPad(times[-1], times[i])        
        #save the figure
        print("Saving: " + save_path + g.name + "_" + s + ".png")
        plt.savefig(save_path + g.name + "_" + s + ".png")
        plt.clf()
        plt.cla()
    #now make the movie
    #strip anything after the .
    s = s.split('.')
    s = len(s[0])
    subprocess.call(["C:\\users\\doug\\desktop\\CM3Dv4\\MakeMovie.bat",
                     save_path + g.name + "_%0" + repr(s) + "d.0.png",
                     save_path + g.name + "_"])
        
def save_3D_gradient_images(save_path, data, name):
    """ Makes an avi of the volume rendered gradient specified by the name
        parameter using visvis
    """
    #load all of the volumes into memory
    vols = []
    #setup the visvis setup to capture all of the images
    f = vv.clf()
    a = vv.gca()
    m = vv.MotionDataContainer(a, interval = 500)
    #loop over all the times
    times = data.get_times()
    mn = 1000000000000
    mx = -1
    #get the min and max range over which to scale the image set
    for i in range(0, len(times)):
        #load the volume
        vol = data.get_gradient_at_time(times[i], name).C
        #get the min and max of these to save
        mx = max(np.max(vol), mx)
        mn = min(np.min(vol), mn)
        #delete the loaded volume for memory issues
        del vol
    #now take screen shots of these images
    for i in range(0, len(times)):
        #load the volume
        vol = data.get_gradient_at_time(times[i], name).C
        print(vol.shape)
        #draw it
        t = vv.volshow(vol)
        t.colormap = vv.CM_HOT
        t.renderstyle = 'iso'
        t.parent = m
        #set the camera filed of view
        a.camera.fov = 60
        a.camera.elevation = 30
        a.camera.zoom = 0.01
        #set the limits to the min and max
        t.clim = (mn, mx)
        #Force the axis to draw now
        a.Draw()
        f.DrawNow()
        t_m.sleep(1)
        #get the zero padded time
        s = ZeroPad(times[-1], times[i])
        #create the save file path
        file_path = save_path + name + "_" + s + ".jpeg"
        #Now get the screenshot
        vv.screenshot(file_path)
        #load this image using PIL, append a color bar showing the min and max
        ss = Image.open(file_path)
        w, h = ss.size
        #make a new font
        font = ImageFont.truetype("arial.ttf", 24)
        color_bar = draw_color_bar(mn, mx, 50, int(.99*h), font)
        #get the colorbar dimension
        w1, h1 = color_bar.size
        #make a new image
        ht = max(h, h1)
        im_final = Image.new("RGB", (w + w1 + 10, ht), "white")
        #paste the other images on to this
        im_final.paste(color_bar, (w + 10, 0))
        im_final.paste(ss, (0, 0))
        #Now finally draw the time and the text on top of this
        draw = ImageDraw.Draw(im_final)
        #get the text to draw
        txt = "Species: " + name + "    Time: " + repr(times[i])
        #gte the position
        pos = (25, 20)
        draw.text(pos, txt, font = font, fill = (0,0,0))
        #save the final file
        im_final.save(file_path)
    #close all the figures
    vv.closeAll()

def save_clustered_data_images(save_path, xs, ys, times, cols, markers):
    """ Saves a series of figures shwoing how the different metrics change as
        a 2D slice through time
        xs - the first metric to plot, contains a list of the values by groups
        ys - the second metric to plot, contains a list of the values by groups
        times - the times over which to plot
        fmt - the format to associated with plotting each group
    """
    #plot the clustered x's and y's through time
    #first find the min and max of each of these data sets through time
    min_x = 1000000000000000000
    max_x = -1
    min_y = 1000000000000000000
    max_y = -1
    for j in range(0, len(xs)):
        for i in range(0, len(times)):
            x = xs[j][i]
            y = ys[j][i]
            max_y = max(max_y, y)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            min_x = min(min_x, x)
    #now we know the min and max, so plot these data
    for i in range(0, len(times)):
        print(times[i])
        for j in range(0, len(xs)):
            plt.xlim(min_x, max_x)
            plt.ylim(min_y, max_y)
            plt.scatter(xs[j][i], ys[j][i], c = cols[j], marker = markers[j])
        #all groups are plotted, so save
        plt.savefig(save_path + repr(i) + ".jpeg")
        plt.clf()
        plt.cla()
            
    

    
    
