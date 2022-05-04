from cosypose.scripts.sequence_script import sequence

def main():
    delta_t_predictions = []
    delta_t_detections = []
    delta_t_renderer = []
    delta_t_network = []
    list_of_objects = ["soup1", "soup3", "soup4", "solo_stairs1", "solo_stairs3", "solo_stairs4", "switch1", "switch3", "switch5", "powerstrip1", "powerstrip4", "powerstrip10"]
    for object in list_of_objects:
        print(object)
        deltap, deltad, deltar, deltan = sequence(object, 3, True)
        delta_t_predictions.append(deltap)
        delta_t_detections.append(deltad)
        delta_t_renderer.append(deltar)
        delta_t_network.append(deltan)
    
    print(delta_t_detections)

    

if __name__ == '__main__':
    main()