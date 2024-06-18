from graphviz import Digraph
import os

# Create a new directed graph
flowchart = Digraph()

# Adding nodes to the flowchart
flowchart.node('A', 'Persiapan')
flowchart.node('B', 'Studi Literatur')
flowchart.node('C', 'Persiapan Bahan (Perangkat Lunak)')
flowchart.node('D', 'Pengambilan Data Simulasi PLECS')
flowchart.node('E', 'Membuat Dataset Simulasi')
flowchart.node('F', 'Membuat Dataset Komponen (Induktor & Kapasitor)')
flowchart.node('G', 'Membangun Model Neural Network')
flowchart.node('H', 'Optimasi Parameter dengan Algoritma')
flowchart.node('I', 'Implementasi Simulasi HIL')
flowchart.node('J', 'Perbandingan Hasil')


# Adding edges to the flowchart
flowchart.edge('A', 'B')
flowchart.edge('A', 'C')
flowchart.edge('B', 'D')
flowchart.edge('C', 'D')
flowchart.edge('D', 'E')
flowchart.edge('D', 'F')
flowchart.edge('E', 'G')
flowchart.edge('F', 'G')
flowchart.edge('G', 'H')
flowchart.edge('H', 'I')
flowchart.edge('I', 'J')

# Specify the correct path for your system
file_path = r'E:\ai-power-converter-1\design'


# Render the graph to a PNG file
flowchart.render(file_path, format='png', cleanup=True)

# Open the file automatically (only works on Windows)
os.startfile(file_path + '.png')