% the program for extended federated learning
% the input-output data is named VehicleData in this program, with
% dimension of N*M*K,N is record number, M is attribute number, K is
% vehicle number.For the content of the M attributes, the first one is
% always 1, and the following 2 to k+1 ones are universal factors, and the
% k+2 to M-1 attributes are local factors, and the last attribute is the
% output (energy consumption)
Prediction_error_record=[];% n*Vehicle_number, to record the prediction error on energy consumption
Omega_record={};% record all the paramerers for all the vehicles, DataRecord*attribute*vehiclenumber
Count_data=size(VehicleData,1);% data number
Attribute_data=size(VehicleData,2)-1;% data attributes
Vehicle_number=size(VehicleData,3);% counting of vehicle number
Omega_overall=0.000000000000001*ones(1,Attribute_data);% initialization of model parameters, with U_P unversal parameters and Attribute_data-U_P local parameters
eta=0.000005;%%%%%%step length for SGD, whose dimension should be consistent with vehicle numbers
Vehicle_specification_symbol=[];% vehicle specification label on whether abnormal(0), extensive(2), or normal(1), or -1 for prediction error less than threshold
for number_v=1:1:Vehicle_number
    Omega_record{1,1,number_v}=Omega_overall;
end

for Num=1:1:Count_data
    Learned__result=zeros(1,Attribute_data);
    %%Learned_integrated_temperal=zeros(1,Attribute_data);
    for Num_veh=1:1:Vehicle_number
        Prediction_error_record(Num,Num_veh)=VehicleData(Num,1:Attribute_data,Num_veh)*[Omega_record{Num,1,Num_veh}]'-VehicleData(Num,(Attribute_data+1),Num_veh);
        data_temporal=VehicleData(Num,:,Num_veh);
        omega_temporal=Omega_record{Num,1,Num_veh};
        [VehicleID,Learned__result] = agent_learning_local(Num_veh,data_temporal,omega_temporal,eta);
        %%Learned_integrated_temperal=Learned_integrated_temperal+Learned__result;
        Omega_record{Num,1,Num_veh}=Omega_record{Num,1,Num_veh}+Learned__result/(norm(Learned__result))/150;
    end
     Parameter_temperal=zeros(1,Attribute_data);
     for Num_veh_j=1:1:Vehicle_number
         Parameter_temperal=Parameter_temperal+Omega_record{Num,1,Num_veh_j};
     end
    for Num_veh_i=1:1:Vehicle_number
        Omega_record{Num+1,1,Num_veh_i}=Parameter_temperal/Vehicle_number;
    end
end
        
