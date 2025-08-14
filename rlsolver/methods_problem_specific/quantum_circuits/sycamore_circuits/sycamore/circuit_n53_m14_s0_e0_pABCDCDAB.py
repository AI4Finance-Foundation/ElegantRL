import cirq
import numpy as np

QUBIT_ORDER = [
    cirq.GridQubit(0, 5),
    cirq.GridQubit(0, 6),
    cirq.GridQubit(1, 4),
    cirq.GridQubit(1, 5),
    cirq.GridQubit(1, 6),
    cirq.GridQubit(1, 7),
    cirq.GridQubit(2, 4),
    cirq.GridQubit(2, 5),
    cirq.GridQubit(2, 6),
    cirq.GridQubit(2, 7),
    cirq.GridQubit(2, 8),
    cirq.GridQubit(3, 2),
    cirq.GridQubit(3, 3),
    cirq.GridQubit(3, 4),
    cirq.GridQubit(3, 5),
    cirq.GridQubit(3, 6),
    cirq.GridQubit(3, 7),
    cirq.GridQubit(3, 8),
    cirq.GridQubit(3, 9),
    cirq.GridQubit(4, 1),
    cirq.GridQubit(4, 2),
    cirq.GridQubit(4, 3),
    cirq.GridQubit(4, 4),
    cirq.GridQubit(4, 5),
    cirq.GridQubit(4, 6),
    cirq.GridQubit(4, 7),
    cirq.GridQubit(4, 8),
    cirq.GridQubit(4, 9),
    cirq.GridQubit(5, 0),
    cirq.GridQubit(5, 1),
    cirq.GridQubit(5, 2),
    cirq.GridQubit(5, 3),
    cirq.GridQubit(5, 4),
    cirq.GridQubit(5, 5),
    cirq.GridQubit(5, 6),
    cirq.GridQubit(5, 7),
    cirq.GridQubit(5, 8),
    cirq.GridQubit(6, 1),
    cirq.GridQubit(6, 2),
    cirq.GridQubit(6, 3),
    cirq.GridQubit(6, 4),
    cirq.GridQubit(6, 5),
    cirq.GridQubit(6, 6),
    cirq.GridQubit(6, 7),
    cirq.GridQubit(7, 2),
    cirq.GridQubit(7, 3),
    cirq.GridQubit(7, 4),
    cirq.GridQubit(7, 5),
    cirq.GridQubit(7, 6),
    cirq.GridQubit(8, 3),
    cirq.GridQubit(8, 4),
    cirq.GridQubit(8, 5),
    cirq.GridQubit(9, 4),
]

CIRCUIT = cirq.Circuit(
    [
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 2)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 9)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 8)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 0)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 1)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.4326562950300605).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-2.225882728378087).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.7293249642087485).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=1.210696502097985).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-1.106519059371497).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=1.7892230375389613).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=0.2119958279956058).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-0.11128338095950507).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.8937947545666063).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-2.9549982284793814).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=1.2842227411644753).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=-1.1227354153032945).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=1.3465899378642412).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-1.781829442911482).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=2.1872907310716982).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-1.9614019632142368).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=1.5928715721087023).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-1.5401880072083034).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-2.5911695330397535).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=2.612251899130335).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=2.404798154102042).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-2.3947316562728576).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-2.365932000948667).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=2.2191901639168874).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-2.4350373161309906).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=3.0221985839799075).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-2.610806420739799).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=2.560219630921752).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=1.785834483295006).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-1.7147844803365624).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=1.4590820918158975).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.7148644518625993).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=1.0568019349008715).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-1.294748318842905).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=2.7682990982443236).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-2.576048471247548).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-1.6256583901850972).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=1.6003028045331185).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-0.8433460214812949).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=0.8404319695057677).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=2.4767110913272044).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.92322176167281).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=1.718893981077632).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-1.8507490475426254).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=2.3200371653684186).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-2.340928713256203).on(cirq.GridQubit(8, 5)),
            cirq.Rz(rads=-0.5750038022730992).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-0.5519833389561652).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5157741664069029, phi=0.5567125777723744).on(
                cirq.GridQubit(0, 6), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5177580142209797, phi=0.4948108578225166).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.6036738621219824, phi=0.47689957001758815).on(
                cirq.GridQubit(1, 7), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5177327089642791, phi=0.5058312223892585).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5253184444076096, phi=0.46557175536519374).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.6141004604574274, phi=0.494343440675308).on(
                cirq.GridQubit(2, 8), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.5476810407275208, phi=0.44290174465702487).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.5237261387830179, phi=0.4696616122846161).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5285844942020295, phi=0.5736654641906893).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5483159975149505, phi=0.4961408893973623).on(
                cirq.GridQubit(3, 9), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.6377079485605028, phi=0.6888985951517526).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.529949914236052, phi=0.48258847574702635).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5280421758407217, phi=0.5109767145462891).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.5120782868771674, phi=0.4815152809861558).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5071938854285831, phi=0.5089276063739265).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(6, 1)
            ),
            cirq.FSimGate(theta=1.5460100224551203, phi=0.5302403303961576).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5166625940397171, phi=0.4517159790427676).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.4597689731864314, phi=0.4214985958536156).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.535649445690472, phi=0.47076284376181704).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(7, 2)
            ),
            cirq.FSimGate(theta=1.5179778495708582, phi=0.5221350266177334).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.4969321270213238, phi=0.4326117171327162).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.5114987201637704, phi=0.4914319343687703).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(8, 3)
            ),
            cirq.FSimGate(theta=1.4908807480930255, phi=0.4886243720131578).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(8, 5)
            ),
            cirq.FSimGate(theta=1.616256999726831, phi=0.501428936283957).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(9, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.3932854391951004).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=2.600059005847074).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=2.1293419565556686).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=2.635214888513154).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=1.0969000068096602).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-0.4141960286421961).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=1.448470803234792).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-1.3477583561986914).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.08664533842437).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-2.147848812337145).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.08243681179982776).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=0.07905051406135266).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=2.8499779533489917).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=2.997967848783354).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-1.4497132543865328).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=1.6756020222439947).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=1.6110403638230537).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-1.5583567989226532).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=1.831437049428291).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-1.8103546833377073).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=0.0716641316809925).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-0.06159763385180828).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=2.6775858065265927).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-2.824327643558372).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.9382895697303493).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-2.3511283018814324).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-2.896580447572063).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=2.8459936577540166).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=2.2114367148817493).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-2.1403867119233055).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=2.1881424887585386).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=2.839260458374346).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-1.4234721207147139).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=1.1855257367726804).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=1.320013738944822).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-1.1277631119480453).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=2.995416784026747).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-3.020772369678725).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=1.686606820477035).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-1.6895208724525625).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=1.609733929520253).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-2.492986383699826).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=-2.481445005407897).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=2.349589938942904).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-2.6928020513607422).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=2.671910503472958).on(cirq.GridQubit(8, 5)),
            cirq.Rz(rads=2.857168426223337).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=2.2990297397269845).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 1)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 9)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 0)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(9, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.2268745424925527).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=0.5571415271071771).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-2.2513994678708658).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=2.2546380725286257).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-0.41246800066795863).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=0.4133873827301695).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=2.3563099677710264).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-0.9989369553991843).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-2.7510095309577114).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-2.930393474388311).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=2.2215976945329947).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-1.9682073694249083).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=2.6009354607665633).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-2.7001293760780083).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=2.5335560654853895).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.39691027011676).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-0.7423813429186499).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=0.7991639946126413).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=2.0345218643948497).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=-2.050938322737726).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-0.6559509011901303).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.8184806876376881).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.4433245645000508).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=2.7662935758886693).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-1.6140341496881252).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=1.5135909743353206).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=2.4170911887166646).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-2.412768992191676).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=0.44574337984494505).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-0.42078376332380696).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-3.0038933332707183).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-2.8323993477171854).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-0.9953713874379053).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=0.910504771397445).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-1.0166585860487407).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=0.8774840291195077).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=2.7788749929698184).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.683035523976411).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5192859850645715, phi=0.49387245572956845).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5120572609259932, phi=0.5211713721957417).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(2, 4)
            ),
            cirq.FSimGate(theta=1.615096254724942, phi=0.7269644142795206).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.5269188098932545, phi=0.5036266469194081).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5179602369002039, phi=0.4914328237769214).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.537427483096926, phi=0.45115204759967975).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.53486366638317, phi=0.46559889697604934).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5194586006330344, phi=0.5068560732280918).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5819908111894527, phi=0.5595875596558442).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.5070579127771144, phi=0.4520319437379419).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.527311970249325, phi=0.4920204543143157).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5139592614725292, phi=0.45916943035876673).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5425204049671604, phi=0.5057437054391503).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.5432564804093452, phi=0.5658780292653538).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.4611081680611278, phi=0.5435451345370095).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.4529959991143482, phi=0.43830813863531964).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.6239619425876082, phi=0.4985036227887135).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.4884560669186362, phi=0.4964777432807688).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.5034760541420213, phi=0.5004774067921798).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(8, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.32256664544673086).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=0.461449424152999).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=1.6662845869106162).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-1.6630459822528558).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-2.0807726116383005).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=2.0816919937005114).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=0.21310624160012637).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=1.1442667707717158).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-3.057055440423247).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-2.6243475649227754).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-1.6842610823270208).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=1.9376514074351068).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-1.9233198015746487).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=1.8241258862632037).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=3.0412260649904477).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.9045802696218184).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-2.053004862643731).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=2.1097875143377225).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-2.3270525120981063).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=2.31063605375523).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-0.19919797967155972).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-1.2752336091562588).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.43179848523061).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=1.7547674966192288).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.902192486573229).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-3.0026356619260337).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=1.9833150852685666).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-1.9789928887435797).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-0.6369338680732861).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.661893484594426).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=1.6098110903736034).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-1.1629184641819208).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-0.9546461682460753).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=0.869779552205615).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=0.825744282325278).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-0.9649188392545094).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=2.356987377884714).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.2611479088913065).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 0)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 1)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 8)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.7296498014579313).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-1.590368513185332).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=2.159690705273027).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-2.144666104810616).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=1.1806041264524705).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-1.058402647956994).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=1.622814550818351).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-3.090826498553735).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=2.0475467080709855).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-2.808240772656996).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=0.0363681235151887).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=0.6267368345703623).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-1.683077723731188).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=1.821251782956388).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=1.889905607894633).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-1.8011305030821063).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=1.191224713999632).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.126642695386768).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=1.9658871892348433).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=-1.9511085272322664).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-1.4198006347182375).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=0.5225414603032448).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-2.7119350351230427).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=1.7532928428796295).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=2.270174449602111).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-2.421631498515731).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-1.9757472607933801).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=2.690718125624965).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-1.5268156554449863).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=1.2787749581888708).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=2.3593119641929743).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-2.420796885290144).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=1.6485284700460934).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-0.9914944601366429).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=1.3965268522136594).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=-1.3773869100671003).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=0.7250694837307181).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-0.7633051520811271).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-2.1782266577776674).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=2.43381058936766).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=1.9783871934256716).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-2.0428219001414405).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=2.616236186508104).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.816811184588105).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=0.3777875775333017).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-0.09226863764678939).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5508555127616375, phi=0.48773645023966805).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(0, 6)
            ),
            cirq.FSimGate(theta=1.4860895179182787, phi=0.49800223593597315).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5268891182960795, phi=0.5146971591948788).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(1, 7)
            ),
            cirq.FSimGate(theta=1.5004518396933153, phi=0.541239891546859).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.5996085979256793, phi=0.5279139399675195).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(2, 8)
            ),
            cirq.FSimGate(theta=1.5354845176224254, phi=0.41898979144044296).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(3, 3)
            ),
            cirq.FSimGate(theta=1.545842827888829, phi=0.533679342490625).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5651524165810975, phi=0.5296573901163858).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.6240366191418867, phi=0.48516108212176406).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(3, 9)
            ),
            cirq.FSimGate(theta=1.6022614099028112, phi=0.5001380228896306).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.574931196238987, phi=0.5236666378689078).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5238301684218176, phi=0.47521120348925566).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5426970250652188, phi=0.5200449092580564).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.4235475054732074, phi=0.5253841271266504).on(
                cirq.GridQubit(5, 0), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.511471063363894, phi=0.4578807555552488).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5371762819242982, phi=0.5674318212304278).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.510414477168897, phi=0.44988262527024675).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.498535212903308, phi=0.637164678333888).on(
                cirq.GridQubit(6, 1), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.507377591132129, phi=0.47869828403704195).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.48836082148729, phi=0.46458301209227065).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.5400981673597602, phi=0.5128416009465753).on(
                cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.5860873970424136, phi=0.4790438939428006).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.5630547528566316, phi=0.48589356877723594).on(
                cirq.GridQubit(8, 3), cirq.GridQubit(8, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.055478127333757).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-3.088425891573232).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-2.224048533450526).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=2.239073133912937).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-1.5717956753246403).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=1.693997153820117).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-1.5534983747823254).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=0.08548642704694132).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=1.851590500689747).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-2.612284565275756).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=-0.6707597723908929).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=1.333864730476444).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-0.4627326605980393).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=0.6009067198232394).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=0.030858809637424045).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=0.05791629517510266).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-0.6573659838051862).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-0.27805199758195087).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-0.15131986471501513).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=0.16609852671759207).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-1.616287932958402).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=0.7190287585434092).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=2.0543843880082964).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-3.0130265802517098).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-1.918109093883396).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=1.7666520449697758).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=1.4266082874857098).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=-0.7116374226541247).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-0.31165755635697856).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=0.06361685910086301).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.420231334847587).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.358746413750417).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.9214866744162733).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-2.7046646228538638).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-2.074440199810285).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=2.093580141956844).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=0.03769063813623674).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-0.075926306486644).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=2.3692232378041176).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-2.113639306214125).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=0.6857578577092225).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-0.7501925644249887).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=2.7570195705710665).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.957594568651068).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=1.5950409783058557).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-1.309522038419343).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 2)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 1)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 0)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 2)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=1.5514777117377587).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-1.471042458257246).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=0.9376872138295304).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-2.505708502040484).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-0.8138828334823582).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=1.4625269352538472).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=1.5465631243549467).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-1.3779143806237855).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-3.056522836696912).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=3.0197041955368213).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=1.4313150179323237).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-1.4458071174911336).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-0.5377106953899506).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-0.2326881725325665).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.14916236480551603).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-0.0049063314348209985).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=0.7506379057684889).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-0.9003639271372892).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-1.8474993478983084).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=1.9987708035250034).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=1.245482758924572).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-1.2148704795940066).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=0.03686637959834638).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=0.0035863783361342882).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-1.8613845453503193).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.1274320925128025).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-1.3490677361559822).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=1.6785360491008614).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=-2.5257109756448184).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=2.10867489793368).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-2.759920821735129).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=2.7202395614573565).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-0.4183020554137258).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=0.9848848466497087).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-0.42166402519488333).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=0.44428590744672997).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-0.77688951203044).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-0.5622182800396907).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=2.348540240114408).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-2.7786467500008385).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.483304234663135, phi=0.4930986258758784).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5160176987076064, phi=0.49850252902921577).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.5338611249160479, phi=0.5011308712767845).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.480179689158691, phi=0.4772322221553844).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5152260103068818, phi=0.49796235787029736).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5284924549164072, phi=0.5160847019657471).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.58661284381037, phi=0.475779832809368).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.5333295850816209, phi=0.44983388105304506).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5158107733607835, phi=0.4663776718737318).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5645151084457722, phi=0.47497942677256283).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.5659881784786247, phi=0.5656290235103623).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.5211879663086973, phi=0.5056110683638391).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5083349422212171, phi=0.49641600818144604).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.5087002777200382, phi=0.44025777694304247).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5658333118222365, phi=0.47264531483343447).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5219378850865568, phi=0.5335829954491795).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.5501487671051402, phi=0.4404117539373896).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.4825325148661492, phi=0.6857341218223484).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.4941963673904604, phi=0.45895108234543025).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.5487430259667763, phi=0.4467898473637848).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(8, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-1.8841634261494544).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=1.9645986796299673).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=2.885627836077404).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=1.8295361828912287).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=0.05530092720528268).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=0.5933431745662068).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-1.2922413166611904).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.4608900603923516).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-2.3332667091615846).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=2.296448068001496).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-1.436526112189684).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=1.4220340126308777).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=2.6035758286461963).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=2.90921061061087).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-1.7401438372662703).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=1.586075141025933).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=0.5464640634289042).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-0.6961900847977063).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=0.7545429458085255).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-0.6032714901818306).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=-1.1636385149103123).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=1.1942507942408778).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-1.071330330465674).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=1.1117830884001547).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.06057801616892).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-1.7945304690064374).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=1.413295584834648).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-1.0838272718897688).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=1.3041394540259699).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-1.7211755317371082).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-1.8989017400046564).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=1.8592204797268828).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-1.0625073171877455).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=1.6290901084237286).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=1.2492236358488604).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-1.2266017535970137).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=1.4019785896279764).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-2.741086381698107).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=-2.3814810782487683).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=1.951374568362338).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 8)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 0)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 8)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(9, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.509738315706713).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-1.3704570274341172).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=2.153407519965878).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-2.138382919503467).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=2.814232306319262).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.6920308278237854).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-1.8455037387446078).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=0.37749179100920927).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-0.5222760825654262).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-0.23841798202058587).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=-2.294693625448355).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=2.957798583533906).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=0.6354176546180561).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.497243595392856).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-1.364784381224447).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=1.4535594860369727).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-2.5347041731578344).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=1.5992861917706982).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-1.5212806562497523).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=1.5360593182523292).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-0.4836060239484059).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.413653150466585).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=2.880099888266729).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=2.4444432266694456).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=2.967608018699071).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-3.119065067612691).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=1.7627479969783801).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=-1.047777132146802).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-2.9279659789460553).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=2.6799252816899397).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.3666169229644929).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=1.3051320018673227).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.3459620391428473).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-1.688928029233397).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=0.7053764684240491).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=-0.6862365262774901).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=1.6612640945005488).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-1.6994997628509552).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=2.012657942111037).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-1.7570740105210447).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=0.809714726290327).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-0.8741494330060942).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=2.8487140428736204).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-3.049289040953621).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-1.9469909861230956).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=2.232509926009607).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5508555127616375, phi=0.48773645023966805).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(0, 6)
            ),
            cirq.FSimGate(theta=1.4860895179182787, phi=0.49800223593597315).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5268891182960795, phi=0.5146971591948788).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(1, 7)
            ),
            cirq.FSimGate(theta=1.5004518396933153, phi=0.541239891546859).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.5996085979256793, phi=0.5279139399675195).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(2, 8)
            ),
            cirq.FSimGate(theta=1.5354845176224254, phi=0.41898979144044296).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(3, 3)
            ),
            cirq.FSimGate(theta=1.545842827888829, phi=0.533679342490625).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5651524165810975, phi=0.5296573901163858).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.6240366191418867, phi=0.48516108212176406).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(3, 9)
            ),
            cirq.FSimGate(theta=1.6022614099028112, phi=0.5001380228896306).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.574931196238987, phi=0.5236666378689078).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5238301684218176, phi=0.47521120348925566).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5426970250652188, phi=0.5200449092580564).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.4235475054732074, phi=0.5253841271266504).on(
                cirq.GridQubit(5, 0), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.511471063363894, phi=0.4578807555552488).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5371762819242982, phi=0.5674318212304278).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.510414477168897, phi=0.44988262527024675).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.498535212903308, phi=0.637164678333888).on(
                cirq.GridQubit(6, 1), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.507377591132129, phi=0.47869828403704195).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.48836082148729, phi=0.46458301209227065).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.5400981673597602, phi=0.5128416009465753).on(
                cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.5860873970424136, phi=0.4790438939428006).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.5630547528566316, phi=0.48589356877723594).on(
                cirq.GridQubit(8, 3), cirq.GridQubit(8, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-1.835566641582535).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=2.9748479298551374).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-2.2177653481433772).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=2.232789948605788).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=3.0777614519881524).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.955559973492676).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=1.914819914780612).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=2.900353444663576).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-1.8617720158534254).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=1.1010779512674134).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=1.6603019765726525).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-0.9971970184871015).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-2.7812280389472868).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.919402098172487).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-2.9976365084230814).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=3.086411613235608).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=3.06856290335228).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=2.27920442244017).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-2.947337326410006).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=2.9621159884125827).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-2.5524825437282335).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=1.6552233693132423).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=2.745534771798109).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=2.5790083431380637).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-2.615542662980356).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=2.464085614066736).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-2.31188697028605).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=3.0268578351176316).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=1.0894927671440904).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-1.337533464400206).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=1.30569755230988).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-1.3671824734070501).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.6642650636665586).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-2.007231053757109).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-1.3832898160206781).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=1.4024297581672371).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-0.8985039726335948).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=0.8602683042831885).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-1.821661362084587).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=2.0772452936745793).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=1.85443032484457).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-1.9188650315603386).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=2.524541714205551).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.7251167122855513).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-2.363365765217333).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=2.648884705103846).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 2)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 8)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 0)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 1)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.0227166097761504).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-1.9422813562956345).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=0.22140408881091742).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-1.7894253770218675).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-1.2788385462138028).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=1.9274826479852925).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=1.7853241660278165).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-1.6166754222966553).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=0.9081670921334216).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.9449857332935103).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=2.1224654017221383).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-2.1369575012809463).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.1713388752567457).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=1.400940007334226).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=1.2394215880812602).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-1.3934902843215973).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.34794598682954).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=1.1982199654607406).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-1.6213046768399444).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=1.772576132466643).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=1.9491995133286792).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-1.9185872339981103).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-2.985345753155011).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=3.0257985110894925).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.79445576726983).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.5284082201073472).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=2.138100109328821).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-1.8086317963839418).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=-2.76447201731769).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=2.3474359396065516).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-1.8300093962726527).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=1.79032813599488).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=2.8363879337053515).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-2.2698051424693713).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=1.2119641546719073).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-1.1893422724200597).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-0.3245001699135024).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-1.0146076221566283).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=0.9536731019204866).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-1.3837796118069186).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.483304234663135, phi=0.4930986258758784).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5160176987076064, phi=0.49850252902921577).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.5338611249160479, phi=0.5011308712767845).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.480179689158691, phi=0.4772322221553844).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5152260103068818, phi=0.49796235787029736).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5284924549164072, phi=0.5160847019657471).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.58661284381037, phi=0.475779832809368).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.5333295850816209, phi=0.44983388105304506).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5158107733607835, phi=0.4663776718737318).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5645151084457722, phi=0.47497942677256283).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.5659881784786247, phi=0.5656290235103623).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.5211879663086973, phi=0.5056110683638391).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5083349422212171, phi=0.49641600818144604).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.5087002777200382, phi=0.44025777694304247).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5658333118222365, phi=0.47264531483343447).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5219378850865568, phi=0.5335829954491795).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.5501487671051402, phi=0.4404117539373896).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.4825325148661492, phi=0.6857341218223484).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.4941963673904604, phi=0.45895108234543025).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.5487430259667763, phi=0.4467898473637848).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(8, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.355402324187839).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=2.435837577668355).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.6812743460835655).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=1.1132530578726154).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=0.5202566399367274).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=0.12838746183476202).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-1.5310023583340602).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.6996511020652214).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.014771330812333616).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.0220473103477552).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.127676495979493).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=2.113184396420685).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.0459812986665984).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=1.2755824307440768).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-3.1287277901530466).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=2.9746590939127096).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=2.645047956026932).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-2.7947739773957316).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=0.5283482747501616).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-0.377076819123463).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=-1.867355269314416).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=1.897967548644985).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=1.950881802287684).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.9104290443532026).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-2.5952622964512297).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.8613098436137125).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-2.0738722606501554).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=2.4033405735950346).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=1.5429004956988415).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-1.9599365734099798).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-2.8288131654671322).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=2.7891319051893593).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=1.9659880008727608).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-1.3994052096367806).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-0.38440454401792934).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=0.4070264262697769).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=0.9495892475110317).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-2.2886970395811623).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=-0.9866139400548483).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=0.5565074301684163).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 9)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 9)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 0)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-1.7330955636302576).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=1.9398691302822326).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-1.3595905672440445).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-0.15903789486672082).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=0.9983480185332675).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-0.31564404036580385).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=1.6005797808823772).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-1.499867333846277).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-2.0133729703407113).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=1.9521694964279346).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.2092282896270623).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=2.370715615488243).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.8505778473320245).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=2.415338342284782).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.7073106232211153).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=2.9331993910785776).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.234561816122234).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=1.2872453810226325).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=0.17343200211866971).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-0.15234963602808804).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=3.127364464427597).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-3.117297966598411).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-0.25478173773634083).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=0.10803990070456493).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.37354651617850365).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=0.2136147516704163).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-1.894523295721189).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=1.843936505903141).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=-3.1338996122268092).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-3.0782356919943297).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=2.1250997343771267).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=2.9023032127557684).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=2.4328195171731437).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.6707659011151748).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-2.145151811969935).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=2.337402438966709).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-0.20565851076299776).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=0.1803029251110182).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-0.1207797111557447).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=0.11786565918021451).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=2.457861535405762).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.942071317594255).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=2.3911948089457944).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-2.5230498754107877).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-1.198546606652119).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=1.177655058764337).on(cirq.GridQubit(8, 5)),
            cirq.Rz(rads=0.10358021090199898).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-1.2305673521312634).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5157741664069029, phi=0.5567125777723744).on(
                cirq.GridQubit(0, 6), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5177580142209797, phi=0.4948108578225166).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.6036738621219824, phi=0.47689957001758815).on(
                cirq.GridQubit(1, 7), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5177327089642791, phi=0.5058312223892585).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5253184444076096, phi=0.46557175536519374).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.6141004604574274, phi=0.494343440675308).on(
                cirq.GridQubit(2, 8), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.5476810407275208, phi=0.44290174465702487).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.5237261387830179, phi=0.4696616122846161).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5285844942020295, phi=0.5736654641906893).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5483159975149505, phi=0.4961408893973623).on(
                cirq.GridQubit(3, 9), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.6377079485605028, phi=0.6888985951517526).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.529949914236052, phi=0.48258847574702635).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5280421758407217, phi=0.5109767145462891).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.5120782868771674, phi=0.4815152809861558).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5071938854285831, phi=0.5089276063739265).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(6, 1)
            ),
            cirq.FSimGate(theta=1.5460100224551203, phi=0.5302403303961576).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5166625940397171, phi=0.4517159790427676).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.4597689731864314, phi=0.4214985958536156).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.535649445690472, phi=0.47076284376181704).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(7, 2)
            ),
            cirq.FSimGate(theta=1.5179778495708582, phi=0.5221350266177334).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.4969321270213238, phi=0.4326117171327162).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.5114987201637704, phi=0.4914319343687703).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(8, 3)
            ),
            cirq.FSimGate(theta=1.4908807480930255, phi=0.4886243720131578).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(8, 5)
            ),
            cirq.FSimGate(theta=1.616256999726831, phi=0.501428936283957).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(9, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=1.7724664194652178).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-1.5656928528132426).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=0.759607559590961).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-2.2782360217017192).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-1.0079670710951056).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=1.6906710492625692).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=0.05988685034802011).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=0.04082559668808017).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=0.7106277561520997).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-0.7718312300648762).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.707297464588221).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=2.8687847904494017).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=0.7639604313656694).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-1.1991999364129118).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.8382972072733033).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=3.0641859751307656).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.8447115551255957).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=1.8973951200259942).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-0.9331644857301312).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=0.9542468518207095).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=-0.6509021786445608).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=0.6609686764737468).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=0.5664355433142667).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-0.7131773803460426).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.1297057374208599).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=0.45745553042806003).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=2.6703217345889136).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-2.720908524406962).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=0.8479855032239847).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-0.7769355002655338).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=1.5221248461973182).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.7779072062440058).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-2.799489702986982).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.561543319044951).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-0.04972065802050806).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=0.24197128501728213).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=1.5754169046046478).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-1.6007724902566276).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=0.964040510151481).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-0.966954562127011).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=1.6285834854416945).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-2.511835939621264).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=3.1294394739035276).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=3.0218907668110653).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=0.8257817206597977).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-0.8466732685475797).on(cirq.GridQubit(8, 5)),
            cirq.Rz(rads=2.178584413048238).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=2.9776137529020836).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 2)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 0)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.2708568396429989).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=0.5131592299567238).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=1.248334748228233).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-1.245096143570473).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=1.7049654478519258).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-1.704046065789715).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=0.2828588164022108).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=1.074514195969634).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=2.8221758365108265).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-2.2203935346772603).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=2.2153145092260527).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-1.9619241841179669).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=0.4960683828611785).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.5952622981726243).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=2.520989694870888).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.3843438995022606).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=2.7322201319514434).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.6754374802574503).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=2.7508049894134636).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=-2.767221447756338).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=0.7640489782325908).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.23848056706041).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-3.109342207061271).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-2.8508740887297).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.983768546653451).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=2.8833253713006393).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-0.4040590142067196).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=0.4083812107316938).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=0.4206106386165622).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-0.3956510220954215).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-3.022742889192781).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-2.813549791795122).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-2.3651057844032337).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=2.2802391683627796).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-2.386392983014069).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=2.2472184260848387).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-2.140859102552003).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=2.2366985715454).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5192859850645715, phi=0.49387245572956845).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5120572609259932, phi=0.5211713721957417).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(2, 4)
            ),
            cirq.FSimGate(theta=1.615096254724942, phi=0.7269644142795206).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.5269188098932545, phi=0.5036266469194081).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5179602369002039, phi=0.4914328237769214).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.537427483096926, phi=0.45115204759967975).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.53486366638317, phi=0.46559889697604934).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5194586006330344, phi=0.5068560732280918).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5819908111894527, phi=0.5595875596558442).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.5070579127771144, phi=0.4520319437379419).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.527311970249325, phi=0.4920204543143157).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5139592614725292, phi=0.45916943035876673).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5425204049671604, phi=0.5057437054391503).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.5432564804093452, phi=0.5658780292653538).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.4611081680611278, phi=0.5435451345370095).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.4529959991143482, phi=0.43830813863531964).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.6239619425876082, phi=0.4985036227887135).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.4884560669186362, phi=0.4964777432807688).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.5034760541420213, phi=0.5004774067921798).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(8, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=0.278584348296274).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=0.5054317213034487).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-1.833449629188484).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=1.8366882338462445).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=2.0849792470214012).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.084059864959194).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=2.2865573929689447).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-0.9291843805970998).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-2.3470555007121945).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=2.9488378025457607).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-1.6779778970200798).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=1.9313682221281656).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=0.1815472763307362).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.28074119164218203).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=3.053792435604949).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.9171466402363215).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=0.7555789696657627).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-0.6987963179717696).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-3.0433356371167157).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=3.0269191787738414).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-1.6191978590942815).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=0.14476627026646227).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-0.7657808426693897).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=1.0887498540580047).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.011258423641035).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=1.9108152482882197).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-1.4787200189876444).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=1.4830422155126222).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-0.6118011268448953).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.636760743366036).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=1.6286606462956676).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-1.1817680201039842).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=0.41508822871925233).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-0.49995484475970997).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=2.1954786792906127).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-2.334653236219843).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=0.9935361662269422).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-0.8976966972335454).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 8)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 9)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 0)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 1)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(9, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-3.121679516517034).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-2.9547322240105807).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=1.191382667470723).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-2.710011129581499).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-2.488819826951536).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-3.1116615020605884).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-2.125349106275088).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=2.2260615533111903).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-1.554700442916623).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=1.4934969690038464).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-1.2793168641643788).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=1.4408041900255597).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-0.06084357094438886).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-0.37439593410285354).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.1500542031989962).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=0.3759429710564586).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-2.1770396121992057).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=2.2297231770996184).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-3.09382435761491).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=3.1149067237054986).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=1.2738247988095885).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-1.2637583009803954).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=0.4489350166677681).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-0.595676853699544).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=1.3097411269483352).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-0.7225798590994152).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-1.6557622540483192).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=1.6051754642302711).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=1.509374329778833).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-1.4383243268203962).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=0.25271051283766255).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.5084928728843643).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=2.891492044597232).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-3.129438428539263).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=0.4058214227450386).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-0.21357079574826443).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-1.8267203200154896).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=1.8013647343635133).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-1.9743193767737497).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=1.971405324798223).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=2.4515783500986146).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.9483545029014024).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=-1.5734951198845515).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=1.441640053419551).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=1.8173823407940866).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-1.8382738886818792).on(cirq.GridQubit(8, 5)),
            cirq.Rz(rads=0.3297748819603683).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-1.4567620231896328).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5157741664069029, phi=0.5567125777723744).on(
                cirq.GridQubit(0, 6), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5177580142209797, phi=0.4948108578225166).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.6036738621219824, phi=0.47689957001758815).on(
                cirq.GridQubit(1, 7), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.5177327089642791, phi=0.5058312223892585).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5253184444076096, phi=0.46557175536519374).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.6141004604574274, phi=0.494343440675308).on(
                cirq.GridQubit(2, 8), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.5476810407275208, phi=0.44290174465702487).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.5237261387830179, phi=0.4696616122846161).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5285844942020295, phi=0.5736654641906893).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5483159975149505, phi=0.4961408893973623).on(
                cirq.GridQubit(3, 9), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.6377079485605028, phi=0.6888985951517526).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.529949914236052, phi=0.48258847574702635).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5280421758407217, phi=0.5109767145462891).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.5120782868771674, phi=0.4815152809861558).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5071938854285831, phi=0.5089276063739265).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(6, 1)
            ),
            cirq.FSimGate(theta=1.5460100224551203, phi=0.5302403303961576).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5166625940397171, phi=0.4517159790427676).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.4597689731864314, phi=0.4214985958536156).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.535649445690472, phi=0.47076284376181704).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(7, 2)
            ),
            cirq.FSimGate(theta=1.5179778495708582, phi=0.5221350266177334).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.4969321270213238, phi=0.4326117171327162).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.5114987201637704, phi=0.4914319343687703).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(8, 3)
            ),
            cirq.FSimGate(theta=1.4908807480930255, phi=0.4886243720131578).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(8, 5)
            ),
            cirq.FSimGate(theta=1.616256999726831, phi=0.501428936283957).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(9, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-3.1221349348275957).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-2.954276805700019).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-1.79136567512381).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=0.2727372130130341).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=2.4792007743896995).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-1.7964967962222342).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-2.497369569674099).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=2.598082016710201).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=0.25195522872801135).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-0.3131587026407878).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=2.645976417128683).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=-2.484489091267502).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.0257738450219662).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.5905343399747238).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=0.8876316798841639).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.6617429120267015).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-0.9022337590486096).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=0.9549173239490222).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=2.334091874003459).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-2.31300950791287).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=1.2026374869734546).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-1.192570989144265).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-0.13728121108984226).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-0.009460625941933642).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-0.8064888733489717).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=1.3936501411978917).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=2.431560692916044).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-2.482147482734092).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=2.4878968683979146).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-2.416846865439478).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=-2.888671239442818).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=1.632888879396116).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=3.025023076768516).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=3.020215846469039).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-2.6006938927354746).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=2.7929445197322487).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-3.086706593322443).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=3.061351007670467).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=2.817580175769489).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-2.820494227745016).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=1.6348666707488422).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-2.5181191249284076).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=0.8109440955542802).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-0.9427991620192806).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-2.1901472267864186).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=2.169255678898626).on(cirq.GridQubit(8, 5)),
            cirq.Rz(rads=1.952389741989869).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-3.0793768832191333).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 9)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 0)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-1.8088774970333825).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=2.592893566633105).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-1.7738773845251252).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=1.7771159891828854).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=0.31638149496535917).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-0.31546211290315895).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-0.4082915673873906).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=1.7656645797592283).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=0.49111408754728103).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=0.11066821428628514).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-1.9755700906626519).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=2.2289604157707377).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-2.299949078833812).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.2007551635223663).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=0.4224058022728556).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-0.2857600069042263).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=1.796025521181612).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-1.7392428694876187).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=2.9895660310863335).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=-3.0059824894292078).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-0.8570128310196984).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.6174187578081209).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.2369529855218069).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=1.559921996910436).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=0.748443525811151).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-0.8488867011639661).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=2.8443477896052194).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-2.840025593080231).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=2.506628160600304).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-2.4816685440791524).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-3.029026074500134).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-2.8072666064877687).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=1.3671062880613754).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-1.45197290410184).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=1.34581908945054).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-1.484993646379781).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=2.502414839453639).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.406575370460242).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5192859850645715, phi=0.49387245572956845).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5120572609259932, phi=0.5211713721957417).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(2, 4)
            ),
            cirq.FSimGate(theta=1.615096254724942, phi=0.7269644142795206).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.5269188098932545, phi=0.5036266469194081).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5179602369002039, phi=0.4914328237769214).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.537427483096926, phi=0.45115204759967975).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.53486366638317, phi=0.46559889697604934).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5194586006330344, phi=0.5068560732280918).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5819908111894527, phi=0.5595875596558442).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.5070579127771144, phi=0.4520319437379419).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.527311970249325, phi=0.4920204543143157).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5139592614725292, phi=0.45916943035876673).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.5425204049671604, phi=0.5057437054391503).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.5432564804093452, phi=0.5658780292653538).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.4611081680611278, phi=0.5435451345370095).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.4529959991143482, phi=0.43830813863531964).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.6239619425876082, phi=0.4985036227887135).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.4884560669186362, phi=0.4964777432807688).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.5034760541420213, phi=0.5004774067921798).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(8, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.358318684972655).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-1.5743026153729325).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=1.188762503564874).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-1.185523898907114).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-2.8096221072716325).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=2.8105414893338363).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=2.977707776758546).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-1.6203347643867119).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.015993751748652585).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=0.6177760535822188).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=2.5129067028686247).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-2.259516377760539).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=2.977564738025727).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-3.0767586533371727).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-1.130808978976603).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=1.2674547743452322).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=1.6917735804355942).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-1.6349909287416011).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=3.0010886283900007).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=-3.017505086732875).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=0.0018639501580004494).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-1.4762955389858199).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.6381700642088397).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=2.961139075597469).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=0.5397148110739458).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-0.640157986426761).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=1.5560584843800136).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-1.5517362878550252).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-2.697818648828626).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.7227782653497776).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=1.634943831603021).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-1.1880512054113375).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.966061463434219).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-3.0509280794746836).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-1.536733393174007).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=1.397558836244766).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=2.6334475314008863).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.5376080624074895).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 8)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 9)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 0)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=1.850003858453061).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-0.7107225701804545).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=2.134557964044429).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-2.119533363582022).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=1.4319315387400522).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-1.3097300602445756).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=0.31591200692573906).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-1.7839239546611234).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-1.9485591472950894).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=1.1878650827090773).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=-3.004693565159407).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-2.6153867839346248).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=1.3077184824862194).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-1.1695444232610193).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=1.4375162657774823).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-1.3487411609649556).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-1.1461202202710616).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=0.2107022388839255).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=0.5835864216556317).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=-0.5688077596530547).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=2.324977808361087).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=3.0609483244035083).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=0.8066487368972908).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.7652909291407148).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-1.2232765811896336).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=1.071819532276014).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=0.4118631559345083).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=0.30310770889709815).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-0.8482316422696741).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=0.6001909450135585).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=0.021967029922279835).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-0.08345195101944825).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-1.844922560746477).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=2.5019565706559277).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-1.368074682944762).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=1.387214625091321).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-1.8133373803695463).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=1.7751017120191366).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=2.0189411274179783).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-1.7633571958279857).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-2.6963026751157138).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=2.6318679683999484).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-2.737037695209419).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=2.5364626971294175).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-2.6381413699127005).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=2.9236603097992138).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5508555127616375, phi=0.48773645023966805).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(0, 6)
            ),
            cirq.FSimGate(theta=1.4860895179182787, phi=0.49800223593597315).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5268891182960795, phi=0.5146971591948788).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(1, 7)
            ),
            cirq.FSimGate(theta=1.5004518396933153, phi=0.541239891546859).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.5996085979256793, phi=0.5279139399675195).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(2, 8)
            ),
            cirq.FSimGate(theta=1.5354845176224254, phi=0.41898979144044296).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(3, 3)
            ),
            cirq.FSimGate(theta=1.545842827888829, phi=0.533679342490625).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5651524165810975, phi=0.5296573901163858).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.6240366191418867, phi=0.48516108212176406).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(3, 9)
            ),
            cirq.FSimGate(theta=1.6022614099028112, phi=0.5001380228896306).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.574931196238987, phi=0.5236666378689078).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5238301684218176, phi=0.47521120348925566).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5426970250652188, phi=0.5200449092580564).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.4235475054732074, phi=0.5253841271266504).on(
                cirq.GridQubit(5, 0), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.511471063363894, phi=0.4578807555552488).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5371762819242982, phi=0.5674318212304278).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.510414477168897, phi=0.44988262527024675).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.498535212903308, phi=0.637164678333888).on(
                cirq.GridQubit(6, 1), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.507377591132129, phi=0.47869828403704195).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.48836082148729, phi=0.46458301209227065).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.5400981673597602, phi=0.5128416009465753).on(
                cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.5860873970424136, phi=0.4790438939428006).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.5630547528566316, phi=0.48589356877723594).on(
                cirq.GridQubit(8, 3), cirq.GridQubit(8, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-1.175832184328872).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=2.315113472601478).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-2.198915792221939).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=2.2139403926843464).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-1.823123087612224).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=1.9453245661077003).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-0.2465958308897207).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-1.2214161168456634).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-0.43548895112376945).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-0.3252051134622427).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=2.370301916283708).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-1.7071969581981534).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=2.829656440364136).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-2.691482381138936).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=0.4832481517545731).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-0.39447304694204627).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=1.6799789504655074).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.615396931852647).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=1.2309809028641965).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=-1.2162022408616195).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=0.9221189311418634).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-1.8193781055568543).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-1.4641993840120442).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=0.5055571917686201).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=1.575341936908349).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-1.7267989858219677).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-0.9610021292421642).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=1.6759729940737709).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-0.9902415695322873).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=0.7422008722761717).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-0.08288640057689278).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=0.021401479479724372).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=0.5719643563762968).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=0.08506965353315366).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=0.6901613353481402).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=-0.6710213932015812).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=2.5760975022365002).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-2.61433317058691).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-1.8279445473915281).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=2.0835284789815205).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-0.9227375809289775).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=0.8583028742132122).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=1.827108145109003).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-2.027683143189005).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-1.672215381427728).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=1.9577343213142395).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 2)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 0)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 1)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.846752003288273).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=2.9271872567687858).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-1.9274452862449074).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=0.3594239980339573).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-2.6737056844081373).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-2.96083552099996).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=2.5016072910464295).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-2.3329585473152683).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=0.23586626426525825).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.2726849054253613).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-2.0872687540880186).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=2.0727766545292035).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-0.7890381076775341).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=0.01863923975501791).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.8780118604379972).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=0.7239431641976743).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.360512357444044).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=1.2107863360752447).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-0.94272066366484).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=1.0939921192915456).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=-2.2228355306385765).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=2.253447809969142).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=0.5143884629440869).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-0.47393570500960536).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-2.087579216408482).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.353626763570965).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=0.033233031424060755).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=0.296235281520822).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=2.8024301648432832).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=3.063719064625161).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=0.9597248801147771).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-0.9994061403925496).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=0.03408728670342143).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=0.5324955045325588).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-0.1703366129073025).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=0.1929584951591501).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=1.0326678564373317).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-2.3717756485074624).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=3.0522569945183093).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=2.800821802774845).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.483304234663135, phi=0.4930986258758784).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5160176987076064, phi=0.49850252902921577).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.5338611249160479, phi=0.5011308712767845).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.480179689158691, phi=0.4772322221553844).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5152260103068818, phi=0.49796235787029736).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5284924549164072, phi=0.5160847019657471).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.58661284381037, phi=0.475779832809368).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.5333295850816209, phi=0.44983388105304506).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5158107733607835, phi=0.4663776718737318).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5645151084457722, phi=0.47497942677256283).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.5659881784786247, phi=0.5656290235103623).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.5211879663086973, phi=0.5056110683638391).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5083349422212171, phi=0.49641600818144604).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.5087002777200382, phi=0.44025777694304247).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5658333118222365, phi=0.47264531483343447).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5219378850865568, phi=0.5335829954491795).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.5501487671051402, phi=0.4404117539373896).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.4825325148661492, phi=0.6857341218223484).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.4941963673904604, phi=0.45895108234543025).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.5487430259667763, phi=0.4467898473637848).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(8, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.514066288876574).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-2.4336310353960613).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-0.5324249710277407).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-1.0355963171832094).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=1.915123778131061).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-1.266479676359572).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-2.247285483352666).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=2.4159342270838273).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=0.6575294970558225).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-0.6943481382159256).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=2.0820576598306495).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-2.0965497593894646).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=2.85490324093378).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=2.6578831983232902).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-1.011294341633775).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=0.8572256453934523).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=2.6576143266414363).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-2.8073403480102357).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-0.15023573842493576).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=0.3015071940516414).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=2.3046797746528362).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-2.274067495322271).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-1.5488524138114137).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=1.5893051717458953).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.2867726872270815).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.0207251400645987).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=0.03099481725460862).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=0.29847349569027415).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=2.2591836207174545).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-2.676219698428593).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=0.6646378653250249).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-0.7043191256027974).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-1.5148966593048918).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.081479450540872).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=0.9978962235612805).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-0.9752743413094329).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-0.4075787788398024).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-0.9315290132303282).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=-3.0851978326526712).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=2.655091322766239).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 4)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 9)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 0)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 1)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 1)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=1.630092372701835).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=-0.4908110844292288).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=2.128274778737282).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=-2.1132501782748783).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=3.065559718606842).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.9433582401113654).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=3.1307790245424165).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=1.6843943349017856).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=1.7648033692480884).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-2.5254974338340865).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=0.947429993056634).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=-0.2843250349710793).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-2.6569714463441194).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.7951455055693053).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-1.817173723341597).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=1.9059488281541235).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=1.4111361997510556).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.346554181138192).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-2.903581423828964).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=2.918360085831541).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-3.022012888048671).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=2.1247537136336767).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=0.11549835310747979).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.0741405453509039).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-0.5258430120926718).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=0.37438596317905404).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=-2.132826893473311).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=2.8477977583049174).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-2.2493819657707412).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=2.0013412685146292).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=2.579223449944397).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-2.6407083710415673).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-1.1474889916497233).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=1.8045230015591744).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-2.059225066734374).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=2.078365008880919).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-0.8771427695997147).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=0.8389071012493013).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-0.07335957987290165).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=0.32894351146289225).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=2.418210164928521).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-2.482644871644297).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-2.504559838843903).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=2.303984840763902).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=1.3202653736104892).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-1.0347464337239742).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.5508555127616375, phi=0.48773645023966805).on(
                cirq.GridQubit(0, 5), cirq.GridQubit(0, 6)
            ),
            cirq.FSimGate(theta=1.4860895179182787, phi=0.49800223593597315).on(
                cirq.GridQubit(1, 4), cirq.GridQubit(1, 5)
            ),
            cirq.FSimGate(theta=1.5268891182960795, phi=0.5146971591948788).on(
                cirq.GridQubit(1, 6), cirq.GridQubit(1, 7)
            ),
            cirq.FSimGate(theta=1.5004518396933153, phi=0.541239891546859).on(
                cirq.GridQubit(2, 5), cirq.GridQubit(2, 6)
            ),
            cirq.FSimGate(theta=1.5996085979256793, phi=0.5279139399675195).on(
                cirq.GridQubit(2, 7), cirq.GridQubit(2, 8)
            ),
            cirq.FSimGate(theta=1.5354845176224254, phi=0.41898979144044296).on(
                cirq.GridQubit(3, 2), cirq.GridQubit(3, 3)
            ),
            cirq.FSimGate(theta=1.545842827888829, phi=0.533679342490625).on(
                cirq.GridQubit(3, 4), cirq.GridQubit(3, 5)
            ),
            cirq.FSimGate(theta=1.5651524165810975, phi=0.5296573901163858).on(
                cirq.GridQubit(3, 6), cirq.GridQubit(3, 7)
            ),
            cirq.FSimGate(theta=1.6240366191418867, phi=0.48516108212176406).on(
                cirq.GridQubit(3, 8), cirq.GridQubit(3, 9)
            ),
            cirq.FSimGate(theta=1.6022614099028112, phi=0.5001380228896306).on(
                cirq.GridQubit(4, 1), cirq.GridQubit(4, 2)
            ),
            cirq.FSimGate(theta=1.574931196238987, phi=0.5236666378689078).on(
                cirq.GridQubit(4, 3), cirq.GridQubit(4, 4)
            ),
            cirq.FSimGate(theta=1.5238301684218176, phi=0.47521120348925566).on(
                cirq.GridQubit(4, 5), cirq.GridQubit(4, 6)
            ),
            cirq.FSimGate(theta=1.5426970250652188, phi=0.5200449092580564).on(
                cirq.GridQubit(4, 7), cirq.GridQubit(4, 8)
            ),
            cirq.FSimGate(theta=1.4235475054732074, phi=0.5253841271266504).on(
                cirq.GridQubit(5, 0), cirq.GridQubit(5, 1)
            ),
            cirq.FSimGate(theta=1.511471063363894, phi=0.4578807555552488).on(
                cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)
            ),
            cirq.FSimGate(theta=1.5371762819242982, phi=0.5674318212304278).on(
                cirq.GridQubit(5, 4), cirq.GridQubit(5, 5)
            ),
            cirq.FSimGate(theta=1.510414477168897, phi=0.44988262527024675).on(
                cirq.GridQubit(5, 6), cirq.GridQubit(5, 7)
            ),
            cirq.FSimGate(theta=1.498535212903308, phi=0.637164678333888).on(
                cirq.GridQubit(6, 1), cirq.GridQubit(6, 2)
            ),
            cirq.FSimGate(theta=1.507377591132129, phi=0.47869828403704195).on(
                cirq.GridQubit(6, 3), cirq.GridQubit(6, 4)
            ),
            cirq.FSimGate(theta=1.48836082148729, phi=0.46458301209227065).on(
                cirq.GridQubit(6, 5), cirq.GridQubit(6, 6)
            ),
            cirq.FSimGate(theta=1.5400981673597602, phi=0.5128416009465753).on(
                cirq.GridQubit(7, 2), cirq.GridQubit(7, 3)
            ),
            cirq.FSimGate(theta=1.5860873970424136, phi=0.4790438939428006).on(
                cirq.GridQubit(7, 4), cirq.GridQubit(7, 5)
            ),
            cirq.FSimGate(theta=1.5630547528566316, phi=0.48589356877723594).on(
                cirq.GridQubit(8, 3), cirq.GridQubit(8, 4)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=-0.9559206985776463).on(cirq.GridQubit(0, 5)),
            cirq.Rz(rads=2.0952019868502525).on(cirq.GridQubit(0, 6)),
            cirq.Rz(rads=-2.1926326069147954).on(cirq.GridQubit(1, 4)),
            cirq.Rz(rads=2.207657207377199).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=2.8264340397005725).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.704232561205096).on(cirq.GridQubit(1, 7)),
            cirq.Rz(rads=-3.0614628485063697).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=1.5934509007709854).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=2.134333839512653).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-2.895027904098665).on(cirq.GridQubit(2, 8)),
            cirq.Rz(rads=-1.5818216419323328).on(cirq.GridQubit(3, 2)),
            cirq.Rz(rads=2.2449266000178874).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=0.5111610620148745).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-0.37298700278968866).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-2.545247166305934).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=2.634022271118461).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-0.8772774695566135).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-0.05814051183052626).on(cirq.GridQubit(3, 9)),
            cirq.Rz(rads=-1.5650365588307942).on(cirq.GridQubit(4, 1)),
            cirq.Rz(rads=1.5798152208333711).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=-0.014075679627968185).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-0.8831834947870263).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-0.7730490002222332).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-0.18559319202119084).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=0.8779083678113891).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-1.0293654167250068).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=1.583687920165655).on(cirq.GridQubit(5, 0)),
            cirq.Rz(rads=-0.8687170553340486).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=0.41090875396878346).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-0.6589494512249026).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-2.640142820599012).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.578657899501845).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-0.1254692127204562).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=0.7825032226299058).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=1.381311719137738).on(cirq.GridQubit(6, 1)),
            cirq.Rz(rads=-1.3621717769911932).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=1.639902891466658).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-1.6781385598170713).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=0.2643561598993518).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-0.00877222830935942).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=0.2459348862063706).on(cirq.GridQubit(7, 2)),
            cirq.Rz(rads=-0.3103695929221324).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=1.5946302887434871).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-1.7952052868234893).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=0.6525631822286719).on(cirq.GridQubit(8, 3)),
            cirq.Rz(rads=-0.3670442423421587).on(cirq.GridQubit(8, 4)),
        ),
        cirq.Moment(
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(0, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(1, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 6)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 8)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 9)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 5)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 9)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 0)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 1)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 7)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 1)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 3)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(8, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
        cirq.Moment(
            cirq.Rz(rads=-2.3755131052498903).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=2.4559483587303994).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=-2.6437284112635204).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=1.0757071230525703).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=-3.1386613971395816).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-2.4958798082685156).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=2.740368332719296).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=-2.571719588988149).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=-2.082629114083989).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=2.0458104729239004).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=-1.3961183702982076).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=1.3816262707393925).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-2.4226662875443203).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=1.6522674196218077).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=0.510572092448772).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=-0.6646407886890948).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=2.8240890571375132).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=-2.9738150785063127).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-0.7165259926064813).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=0.8677974482331727).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=-1.5191187762344747).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=1.5497310555650365).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=-2.5078236698092695).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=2.548276427743751).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=2.5682610962116676).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=-2.302213549049185).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=-2.762784430270724).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=3.0922527432156066).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=2.5636691231704027).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-2.9807052008815553).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=1.8896363055772518).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=-1.9293175658550261).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=-2.9944080313570858).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-2.7221944845865202).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=1.4632915669594873).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=-1.4406696847076397).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=1.4850571985542622).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-2.824164990624393).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=1.6573898563243876).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=-2.0874963662108197).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            cirq.FSimGate(theta=1.483304234663135, phi=0.4930986258758784).on(
                cirq.GridQubit(1, 5), cirq.GridQubit(1, 6)
            ),
            cirq.FSimGate(theta=1.5160176987076064, phi=0.49850252902921577).on(
                cirq.GridQubit(2, 4), cirq.GridQubit(2, 5)
            ),
            cirq.FSimGate(theta=1.5338611249160479, phi=0.5011308712767845).on(
                cirq.GridQubit(2, 6), cirq.GridQubit(2, 7)
            ),
            cirq.FSimGate(theta=1.480179689158691, phi=0.4772322221553844).on(
                cirq.GridQubit(3, 3), cirq.GridQubit(3, 4)
            ),
            cirq.FSimGate(theta=1.5152260103068818, phi=0.49796235787029736).on(
                cirq.GridQubit(3, 5), cirq.GridQubit(3, 6)
            ),
            cirq.FSimGate(theta=1.5284924549164072, phi=0.5160847019657471).on(
                cirq.GridQubit(3, 7), cirq.GridQubit(3, 8)
            ),
            cirq.FSimGate(theta=1.58661284381037, phi=0.475779832809368).on(
                cirq.GridQubit(4, 2), cirq.GridQubit(4, 3)
            ),
            cirq.FSimGate(theta=1.5333295850816209, phi=0.44983388105304506).on(
                cirq.GridQubit(4, 4), cirq.GridQubit(4, 5)
            ),
            cirq.FSimGate(theta=1.5158107733607835, phi=0.4663776718737318).on(
                cirq.GridQubit(4, 6), cirq.GridQubit(4, 7)
            ),
            cirq.FSimGate(theta=1.5645151084457722, phi=0.47497942677256283).on(
                cirq.GridQubit(4, 8), cirq.GridQubit(4, 9)
            ),
            cirq.FSimGate(theta=1.5659881784786247, phi=0.5656290235103623).on(
                cirq.GridQubit(5, 1), cirq.GridQubit(5, 2)
            ),
            cirq.FSimGate(theta=1.5211879663086973, phi=0.5056110683638391).on(
                cirq.GridQubit(5, 3), cirq.GridQubit(5, 4)
            ),
            cirq.FSimGate(theta=1.5083349422212171, phi=0.49641600818144604).on(
                cirq.GridQubit(5, 5), cirq.GridQubit(5, 6)
            ),
            cirq.FSimGate(theta=1.5087002777200382, phi=0.44025777694304247).on(
                cirq.GridQubit(5, 7), cirq.GridQubit(5, 8)
            ),
            cirq.FSimGate(theta=1.5658333118222365, phi=0.47264531483343447).on(
                cirq.GridQubit(6, 2), cirq.GridQubit(6, 3)
            ),
            cirq.FSimGate(theta=1.5219378850865568, phi=0.5335829954491795).on(
                cirq.GridQubit(6, 4), cirq.GridQubit(6, 5)
            ),
            cirq.FSimGate(theta=1.5501487671051402, phi=0.4404117539373896).on(
                cirq.GridQubit(6, 6), cirq.GridQubit(6, 7)
            ),
            cirq.FSimGate(theta=1.4825325148661492, phi=0.6857341218223484).on(
                cirq.GridQubit(7, 3), cirq.GridQubit(7, 4)
            ),
            cirq.FSimGate(theta=1.4941963673904604, phi=0.45895108234543025).on(
                cirq.GridQubit(7, 5), cirq.GridQubit(7, 6)
            ),
            cirq.FSimGate(theta=1.5487430259667763, phi=0.4467898473637848).on(
                cirq.GridQubit(8, 4), cirq.GridQubit(8, 5)
            ),
        ),
        cirq.Moment(
            cirq.Rz(rads=2.0428273908381875).on(cirq.GridQubit(1, 5)),
            cirq.Rz(rads=-1.9623921373576785).on(cirq.GridQubit(1, 6)),
            cirq.Rz(rads=0.18385815399087235).on(cirq.GridQubit(2, 4)),
            cirq.Rz(rads=-1.7518794422018225).on(cirq.GridQubit(2, 5)),
            cirq.Rz(rads=2.3800794908625056).on(cirq.GridQubit(2, 6)),
            cirq.Rz(rads=-1.7314353890910164).on(cirq.GridQubit(2, 7)),
            cirq.Rz(rads=-2.4860465250255466).on(cirq.GridQubit(3, 3)),
            cirq.Rz(rads=2.6546952687566936).on(cirq.GridQubit(3, 4)),
            cirq.Rz(rads=2.976024875405084).on(cirq.GridQubit(3, 5)),
            cirq.Rz(rads=-3.012843516565173).on(cirq.GridQubit(3, 6)),
            cirq.Rz(rads=1.3909072760408385).on(cirq.GridQubit(3, 7)),
            cirq.Rz(rads=-1.4053993755996537).on(cirq.GridQubit(3, 8)),
            cirq.Rz(rads=-1.7946538863790167).on(cirq.GridQubit(4, 2)),
            cirq.Rz(rads=1.024255018456497).on(cirq.GridQubit(4, 3)),
            cirq.Rz(rads=-2.399878294520544).on(cirq.GridQubit(4, 4)),
            cirq.Rz(rads=2.2458095982802213).on(cirq.GridQubit(4, 5)),
            cirq.Rz(rads=-1.526987087940121).on(cirq.GridQubit(4, 6)),
            cirq.Rz(rads=1.3772610665713216).on(cirq.GridQubit(4, 7)),
            cirq.Rz(rads=-0.37643040948330864).on(cirq.GridQubit(4, 8)),
            cirq.Rz(rads=0.5277018651100001).on(cirq.GridQubit(4, 9)),
            cirq.Rz(rads=1.600963020248731).on(cirq.GridQubit(5, 1)),
            cirq.Rz(rads=-1.570350740918169).on(cirq.GridQubit(5, 2)),
            cirq.Rz(rads=1.4733597189419427).on(cirq.GridQubit(5, 3)),
            cirq.Rz(rads=-1.4329069610074612).on(cirq.GridQubit(5, 4)),
            cirq.Rz(rads=-2.3690676253930683).on(cirq.GridQubit(5, 5)),
            cirq.Rz(rads=2.635115172555551).on(cirq.GridQubit(5, 6)),
            cirq.Rz(rads=2.827012278949393).on(cirq.GridQubit(5, 7)),
            cirq.Rz(rads=-2.4975439660045105).on(cirq.GridQubit(5, 8)),
            cirq.Rz(rads=2.4979446623903208).on(cirq.GridQubit(6, 2)),
            cirq.Rz(rads=-2.9149807401014733).on(cirq.GridQubit(6, 3)),
            cirq.Rz(rads=-0.2652735601374516).on(cirq.GridQubit(6, 4)),
            cirq.Rz(rads=0.2255922998596791).on(cirq.GridQubit(6, 5)),
            cirq.Rz(rads=1.5135986587556154).on(cirq.GridQubit(6, 6)),
            cirq.Rz(rads=-0.9470158675196352).on(cirq.GridQubit(6, 7)),
            cirq.Rz(rads=-0.6357319563055093).on(cirq.GridQubit(7, 3)),
            cirq.Rz(rads=0.6583538385573569).on(cirq.GridQubit(7, 4)),
            cirq.Rz(rads=-0.8599681209567329).on(cirq.GridQubit(7, 5)),
            cirq.Rz(rads=-0.4791396711133978).on(cirq.GridQubit(7, 6)),
            cirq.Rz(rads=-1.6903306944587497).on(cirq.GridQubit(8, 4)),
            cirq.Rz(rads=1.2602241845723174).on(cirq.GridQubit(8, 5)),
        ),
        cirq.Moment(
            (cirq.X ** 0.5).on(cirq.GridQubit(0, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(0, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 5)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(1, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(1, 7)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(2, 6)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 7)),
            (cirq.X ** 0.5).on(cirq.GridQubit(2, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 3)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 4)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(3, 6)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(3, 8)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(3, 9)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 1)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 2)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 4)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 5)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 6)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(4, 7)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(4, 8)),
            (cirq.X ** 0.5).on(cirq.GridQubit(4, 9)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 0)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 1)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 2)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 3)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(5, 4)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 5)
            ),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 6)),
            (cirq.X ** 0.5).on(cirq.GridQubit(5, 7)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(5, 8)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 1)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 2)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 3)
            ),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(6, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(6, 7)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 2)),
            (cirq.X ** 0.5).on(cirq.GridQubit(7, 3)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 4)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(7, 5)),
            cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(
                cirq.GridQubit(7, 6)
            ),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 3)),
            (cirq.X ** 0.5).on(cirq.GridQubit(8, 4)),
            (cirq.Y ** 0.5).on(cirq.GridQubit(8, 5)),
            (cirq.X ** 0.5).on(cirq.GridQubit(9, 4)),
        ),
    ]
)

