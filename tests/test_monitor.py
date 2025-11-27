from monitor import NeuralCertificateMonitor
from monitor.adapters import SablasDrone

def test_base():
    drone = SablasDrone()
    m = NeuralCertificateMonitor(drone)
    m.run_till_done()
    m.print_outputs()
