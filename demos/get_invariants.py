#coding=utf8

################################################################################
###                                                                          ###
### Created by Alice Peyraut, 2023-2024                                      ###
###                                                                          ###
### École Polytechnique, Palaiseau, France                                   ###
###                                                                          ###
################################################################################

import dolfin
import numpy

import dolfin_mech as dmech

################################################################################

def get_invariants(
        U_inhal=None,
        U_exhal=None,
        mesh=None,
        lognorm=True):

    ###################################### Retrieving the displacement field ###

    U_tot = U_inhal.copy(deepcopy=True)
    U_tot.vector().set_local(U_inhal.vector()[:] - U_exhal.vector()[:])

    ############################################### Defining function spaces ###

    dolfin.ALE.move(mesh, U_exhal)

    fe_u = dolfin.VectorElement(
        family="CG",
        cell=mesh.ufl_cell(),
        degree=1)
    U_fs = dolfin.FunctionSpace(mesh, fe_u)
    
    sfoi_fe = dolfin.FiniteElement(
        family="DG",
        cell=mesh.ufl_cell(),
        degree=0)
    sfoi_fs = dolfin.FunctionSpace(
        mesh,
        sfoi_fe)
    
    U_tot_mesh=dolfin.Function(U_fs)
    U_tot_mesh.vector().set_local(U_tot.vector()[:]) ### only necessary because using directly U_tot causes dolfin errors for some reason

    #################################################### Defining kinematics ###
    
    kinematics_new = dmech.Kinematics(U=U_tot_mesh, U_old=None, Q_expr=None)

    ########################### Retrieving the invariants values on the mesh ###
    
    J_tot_field = kinematics_new.J
    J_tot_proj = dolfin.project(J_tot_field, sfoi_fs)
    J_tot = J_tot_proj.vector().get_local()

    Ic_tot_field = 1/3*kinematics_new.IC
    Ic_tot_proj = dolfin.project(Ic_tot_field, sfoi_fs)
    Ic_tot = Ic_tot_proj.vector().get_local()

    IIc_tot_field = 1/3*kinematics_new.IIC
    IIc_tot_proj = dolfin.project(IIc_tot_field, sfoi_fs)
    IIc_tot = IIc_tot_proj.vector().get_local()

    ############################# Splitting the mesh into 10 different zones ###

    domains_zones = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
    domains_zones.set_all(10)
    tol =1e-14
    zmin = mesh.coordinates()[:, 2].min()
    zmax = mesh.coordinates()[:, 2].max()
    zones = 10
    delta_z = (zmax-zmin)/(zones+1)
    subdomain_lst = []
    subdomain_lst.append(dolfin.CompiledSubDomain("x[2] <= z1 + tol",  z1=zmin+delta_z, tol=tol))
    subdomain_lst[0].mark(domains_zones, 10)
    for zone_ in range(0, zones-1):
        subdomain_lst.append(dolfin.CompiledSubDomain(" x[2] >= z1 - tol",  z1=zmin+delta_z*(zone_+1), tol=tol))
        subdomain_lst[zone_+1].mark(domains_zones, 10-(zone_+1))

    ###################### Computing the invariants for each of the 10 zones ###
    
    results_zones = {}

    if not lognorm:
        results_zones["zone"] = []
        results_zones["J"] = []
        results_zones["I1"] = []
        results_zones["I2"] = []
        for i in range(1, zones+1):
            results_zones["zone"].append(i)
            marked_cells = dolfin.SubsetIterator(domains_zones, i)
            J_lst = []
            I1_lst = []
            I2_lst = []
            for cell in marked_cells:
                J_lst.append(J_tot[cell.index()])
                I1_lst.append(Ic_tot[cell.index()])
                I2_lst.append(IIc_tot[cell.index()])
            J_average = numpy.average(J_lst)
            I1_average = numpy.average(I1_lst)
            I2_average = numpy.average(I2_lst)

            results_zones["J"].append(J_average)
            results_zones["I1"].append(I1_average)
            results_zones["I2"].append(I2_average)

        return(results_zones)

    else:

        ########################################### Computing mu(invariants) ###
        
        mu_J_tot, mu_Ic_tot, mu_IIc_tot = 0, 0, 0
        
        number_cells = 0
        for cell in range(mesh.num_cells()):
            number_cells += 1
            mu_J_tot += numpy.log(J_tot[cell])
            mu_Ic_tot += numpy.log(Ic_tot[cell])
            mu_IIc_tot += numpy.log(IIc_tot[cell])
        mu_J_tot /= number_cells
        mu_Ic_tot /= number_cells
        mu_IIc_tot /= number_cells

        ######################################## Computing sigma(invariants) ###
        
        sigma_J_tot, sigma_Ic_tot, sigma_IIc_tot = 0, 0, 0

        compteur = 0
        for cell in range(mesh.num_cells()):
            sigma_J_tot += (numpy.log(J_tot[cell])-mu_J_tot)*(numpy.log(J_tot[cell])-mu_J_tot)
            sigma_Ic_tot += (numpy.log(Ic_tot[cell])-mu_Ic_tot)*(numpy.log(Ic_tot[cell])-mu_Ic_tot)
            sigma_IIc_tot += (numpy.log(IIc_tot[cell])-mu_IIc_tot)*(numpy.log(IIc_tot[cell])-mu_IIc_tot)
            compteur += 1
        sigma_J_tot /= number_cells
        sigma_J_tot = sigma_J_tot**(1/2)
        sigma_Ic_tot /= number_cells
        sigma_Ic_tot = sigma_Ic_tot**(1/2)
        sigma_IIc_tot /= number_cells
        sigma_IIc_tot = sigma_IIc_tot**(1/2)

        results_zones["zone"] = []
        results_zones["J^"] = []
        results_zones["I1^"] = []
        results_zones["I2^"] = []

        for i in range(1, zones+1):
            results_zones["zone"].append(i)
            marked_cells = dolfin.SubsetIterator(domains_zones, i)
            J_lst = []
            I1_lst = []
            I2_lst = []
            for cell in marked_cells:
                J_lst.append(J_tot[cell.index()])
                I1_lst.append(Ic_tot[cell.index()])
                I2_lst.append(IIc_tot[cell.index()])
            J_average = numpy.average(J_lst)
            J_chapeau = ((numpy.log(J_average)-mu_J_tot)/sigma_J_tot)
            I1_average = numpy.average(I1_lst)
            I1_chapeau = ((numpy.log(I1_average)-mu_Ic_tot)/sigma_Ic_tot)
            I2_average = numpy.average(I2_lst)
            I2_chapeau = ((numpy.log(I2_average)-mu_IIc_tot)/sigma_IIc_tot)

            results_zones["J^"].append(J_chapeau)
            results_zones["I1^"].append(I1_chapeau)
            results_zones["I2^"].append(I2_chapeau)

        return(results_zones)
