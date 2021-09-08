function [VehicleID,Learned_result] = agent_learning_local(VID,data,Omega,eta)
    VehicleID=VID;
    n=size(data,1);
    all_attribute=size(data,2)-1;
    for Learned_result_attribute=1:1:all_attribute
        Learned_result(1,Learned_result_attribute)=eta*data(n,Learned_result_attribute)*(data(n,(all_attribute+1))-data(n,1:all_attribute)*Omega');
    end
    return;
