import numpy as np
import pyparticle as pp

def compute_field_lines(Bfield, nperiods=200, batch_size=8, magnetic_axis_radius=1, max_thickness=0.5, delta=0.01, steps_per_period=100):

    largest = [0.]

    def rhs(phi, rz):
        nparticles = rz.shape[0]//2
        while phi >= np.pi:
            phi -= 2*np.pi
        rz = rz.reshape((nparticles, 2))
        rphiz = np.zeros((nparticles, 3))
        rphiz[:, 0] = rz[:, 0]
        rphiz[:, 1] = phi
        rphiz[:, 2] = rz[:, 1]
        Brphiz = np.zeros((nparticles, 3))
        for i in range(nparticles):
            Brphiz[i, :] = Bfield.B(rphiz[i, 0], rphiz[i, 1], rphiz[i, 2])
        rhs_rz = np.zeros(rz.shape)
        rhs_rz[:, 0] = rphiz[:, 0] * Brphiz[:, 0]/Brphiz[:, 1]
        rhs_rz[:, 1] = rphiz[:, 0] * Brphiz[:, 2]/Brphiz[:, 1]
        return rhs_rz.flatten()


    from scipy.integrate import solve_ivp, RK45, OdeSolution
    from math import pi

    res = []
    nt = int(steps_per_period * nperiods)
    tspan = [0, 2*pi*nperiods]
    t_eval = np.linspace(0, tspan[-1], nt+1)
    i = 0
    while (i+1)*batch_size*delta < max_thickness:
        y0 = np.zeros((batch_size, 2))
        y0[:, 0] = np.linspace(magnetic_axis_radius + i*batch_size*delta, magnetic_axis_radius+(i+1)*batch_size*delta, batch_size, endpoint=False)
        t = tspan[0]
        solver = RK45(rhs, tspan[0], y0.flatten(), tspan[-1], rtol=1e-9, atol=1e-09)
        ts = [0]
        denseoutputs = []
        while t < tspan[-1]:
            solver.step()
            if solver.t < t + 1e-10: # no progress --> abort
                break
            t = solver.t
            ts.append(solver.t)
            denseoutputs.append(solver.dense_output())
        if t >= tspan[1]:
            odesol = OdeSolution(ts, denseoutputs)
            res.append(odesol(t_eval))
            print(y0[0, 0], "to", y0[-1, 0], "-> success")
        else:
            print(y0[0, 0], "to", y0[-1, 0], "-> fail")
        #     break
        i += 1

    nparticles = len(res) * batch_size

    rphiz = np.zeros((nparticles, nt, 3))
    xyz = np.zeros((nparticles, nt, 3))
    phi_no_mod = t_eval.copy()
    for i in range(nt):
        while t_eval[i] >= np.pi:
            t_eval[i] -= 2*np.pi
        rphiz[:, i, 1] = t_eval[i]
    for j in range(len(res)):
        for i in range(nt):
            rz = res[j][:, i].reshape((batch_size, 2))
            rphiz[j*batch_size:(j+1)*batch_size, i, 0] = rz[:, 0]
            rphiz[j*batch_size:(j+1)*batch_size, i, 2] = rz[:, 1]
    for i in range(nt):
        for j in range(nparticles):
            xyz[j, i, :] = pp.cyl_to_cart(rphiz[j, i, :])


    # absB = np.zeros((nparticles, nt))
    # tmp = np.zeros((nt, 3))
    # for j in range(nparticles):
    #     tmp[:] = 0
    #     cpp.biot_savart_B_only(xyz[j, :, :], gammas, dgamma_by_dphis, biotsavart.coil_currents, tmp)
    #     absB[j, :] = np.linalg.norm(tmp, axis=1)

    return rphiz, xyz#, absB, phi_no_mod[:-1]
