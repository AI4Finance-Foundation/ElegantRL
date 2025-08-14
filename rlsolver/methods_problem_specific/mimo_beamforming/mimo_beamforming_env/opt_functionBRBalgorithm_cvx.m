function [bestFeasible,Woptimal,Whistory,totalNbrOfEvaluations,bounds,boxes] = functionBRBalgorithm_cvx(H,D,Qsqrt,q,boxesLowerCorners,boxesUpperCorners,weights,delta,epsilon,maxIterations,maxFuncEvaluations,localFeasible,problemMode,saveBoxes)
%Maximizes the weighted sum rate or weighted proportional fairness using
%the Branch-Reduce-and-Bound (BRB) algorithm in Algorithm 3. Both problems
%are non-convex and NP-hard in general, thus the computational complexity
%scales exponentially with the number of users, Kr. This implementation is
%not recommend for more than Kr=6 users.
%
%The references to theorems and equations refer to the following book:
%
%Emil Bj�rnson, Eduard Jorswieck, �Optimal Resource Allocation in
%Coordinated Multi-Cell Systems,� Foundations and Trends in Communications
%and Information Theory, vol. 9, no. 2-3, pp. 113-381, 2013.
%
%This is version 1.2. (Last edited: 2014-03-26)
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%original article listed above.
%
%The implementation utilizes and requires CVX: http://cvxr.com/
%
%
%INPUT:
%H             = Kr x Kt*Nt matrix with row index for receiver and column
%                index transmit antennas
%D             = Kt*Nt x Kt*Nt x Kr diagonal matrix. Element (j,j,k) is one 
%                if j:th antenna can transmit to user k and zero otherwise
%Qsqrt         = N x N x L matrix with matrix-square roots of the L 
%              weighting matrices for the L power constraints
%q             = Limits of the L power constraints
%boxesLowerCorners = Kr x 1 vector with lower corner of an initial box that
%                    covers the rate region
%boxesUpperCorners = Kr x 1 vector with upper corner of an initial box that
%                    covers the rate region
%weights       = Kr x 1 vector with positive weights for each user
%delta         = Accuracy of the line-search in FPO subproblems 
%                (see functionFairnessProfile() for details
%epsilon       = Accuracy of the final value of the utility
%maxIterations = Maximal number of outer iterations of the algorithm
%maxFuncEvaluations = Maximal number of convex feasibility subproblems to
%                     be solved
%localFeasible = (Optional) Kr x 1 vector with any feasible solution
%problemMode   = (Optional) Weighted sum rate is given by mode==1 (default)
%                 Weighted proportional fairness is given by mode==2
%saveBoxes     = (Optional) Saves and return the set of boxes from each
%                 iteration of the algorithm if saveBoxes==1
%
%OUTPUT:
%bestFeasible          = The best feasible solution found by the algorithm
%Woptimal              = Kt*Nt x Kr matrix with beamforming that achieves bestFeasible
%totalNbrOfEvaluations = Vector with number of times that the convex 
%                        subproblem was solved at each iteration of the
%                        algorithm
%bounds                = Matrix where first/second column gives the global 
%                        lower/upper bound at each iteration of the algorithm
%boxes                 = Cell array where boxes{k}.lowerCorners and
%                        boxes{k}.upperCorners contain the corners of the
%                        boxes at the end of iteration k.



Kr = size(H,1);  %Number of users (in total)
Nantennas = size(H,2);
I = eye(Kr); %Kr x Kr identity matrix


%Initialize the best feasible solution as origin or point given by input
if nargin < 12
    bestFeasible = zeros(Kr,1);
else
    bestFeasible = localFeasible;
end


%If problemMode has not been specified: Select weighted sum rate
if nargin < 13
    problemMode = 1;
end

%If saveBoxes has not been specified: Do not save and return set of boxes
if nargin < 14
    saveBoxes = 0;
end
boxes{1}.lowerCorners=zeros(Kr,1);
boxes{1}.upperCorners=zeros(Kr,1);


%Pre-allocation of matrices for storing lower/upper bounds on optimal
%utility and the number of times the convex subproblem (power minimization 
%under QoS requirements) is solved.
lowerBound = zeros(maxIterations,1);
upperBound = zeros(maxIterations,1);
totalNbrOfEvaluations = zeros(maxIterations,1);
Whistory = zeros(maxIterations,Nantennas,Kr);
kopt = 1;

%Initialize current best value (cbv) and the current upper bounds (cub),
%where the latter is the potential system utility in each vertex.
if problemMode == 1 %Weighted sum rate
    cbv = weights'*localFeasible;
    cub = weights'*boxesUpperCorners;
elseif problemMode == 2 %Weighted proportional fairness
    cbv = geomean_weighted(localFeasible,weights);
    cub = geomean_weighted(boxesUpperCorners,weights);
end


%Initialize matrix for storing optimal beamforming
Woptimal = zeros(size(H'));



%Iteration of the BRB algorithm. Continue until termination by solution
%accuracy, maximum number of iterations or subproblem solutions 
for k = 1:maxIterations
%     disp(k)
    
    
    %Step 1 of BRB algorithm: Branch
    [~,ind] = max(cub); %Select box with current global upper bound
    
    [len,dim] = max(boxesUpperCorners(:,ind)-boxesLowerCorners(:,ind)); %Find the longest side
    
    %Divide the box into two disjoint subboxes
    newBoxesLowerCorners = [boxesLowerCorners(:,ind) boxesLowerCorners(:,ind)+I(:,dim)*len/2];
    newBoxesUpperCorners = [boxesUpperCorners(:,ind)-I(:,dim)*len/2 boxesUpperCorners(:,ind)];
    
    %Set local lower and upper bounds using Lemma 2.9
    if min(localFeasible(:,ind)>=newBoxesLowerCorners(:,2)) == 1
        point = localFeasible(:,ind)-newBoxesUpperCorners(:,1);
        point(point<0) = 0;
        localFeasibleNew = [localFeasible(:,ind)-point localFeasible(:,ind)];
    else
        localFeasibleNew = [localFeasible(:,ind) localFeasible(:,ind)];
    end
    
    
    %Step 2 of BRB algorithm: Reduce
    
    %Reduction if the two new boxes based on Lemma 2.10
    if problemMode == 1
        
        %Reduction for weighted sum rate is given by Example 2.11
        cubNew = min([weights'*newBoxesUpperCorners; cub(ind)*ones(1,2)],[],1);
        
        newBoxesLowerCornersReduced = zeros(size(newBoxesLowerCorners));
        for m = 1:Kr
            nu = (weights'*newBoxesUpperCorners-cbv)./(weights(m)*(newBoxesUpperCorners(m,:)-newBoxesLowerCorners(m,:)));
            nu(nu>1) = 1;
            newBoxesLowerCornersReduced(m,:) = (1-nu).*newBoxesUpperCorners(m,:)+nu.*newBoxesLowerCorners(m,:);
        end
        
        newBoxesUpperCornersReduced = zeros(size(newBoxesUpperCorners));
        for m = 1:Kr
            mu = (cubNew-weights'*newBoxesLowerCornersReduced)./(weights(m)*(newBoxesUpperCorners(m,:)-newBoxesLowerCornersReduced(m,:)));
            mu(mu>1) = 1;
            newBoxesUpperCornersReduced(m,:) = (1-mu).*newBoxesLowerCornersReduced(m,:)+mu.*newBoxesUpperCorners(m,:);
        end
        
    elseif problemMode == 2
        
        %Reduction for weighted proportional fairness is given by Example 2.12
        cubNew = min([geomean_weighted(newBoxesUpperCorners,weights); cub(ind)*ones(1,2)],[],1);
        
        newBoxesLowerCornersReduced = zeros(size(newBoxesLowerCorners));
        for m = 1:Kr
            nu = (1-(cbv^(1/weights(m)))./(prod(newBoxesUpperCorners.^(repmat(weights,[1 2])/weights(m)),1))) .* newBoxesUpperCorners(m,:) ./(newBoxesUpperCorners(m,:)-newBoxesLowerCorners(m,:));
            nu(nu>1) = 1;
            nu(isnan(nu)) = 1;
            newBoxesLowerCornersReduced(m,:) = (1-nu).*newBoxesUpperCorners(m,:)+nu.*newBoxesLowerCorners(m,:);
        end
        
        newBoxesUpperCornersReduced = zeros(size(newBoxesUpperCorners));
        for m = 1:Kr
            mu = ((cubNew.^(1/weights(m)))./prod(newBoxesLowerCornersReduced.^(repmat(weights,[1 2])/weights(m)),1)-1) .* newBoxesLowerCornersReduced(m,:) ./ (newBoxesUpperCorners(m,:)-newBoxesLowerCorners(m,:));
            mu(mu>1) = 1;
            mu(isnan(mu)) = 1;
            newBoxesUpperCornersReduced(m,:) = (1-mu).*newBoxesLowerCornersReduced(m,:)+mu.*newBoxesUpperCorners(m,:);
        end
        
    end
    
    %Update the two new boxes with the reduced versions
    newBoxesLowerCorners = newBoxesLowerCornersReduced;
    newBoxesUpperCorners = newBoxesUpperCornersReduced;
    
    
    %Step 3 of BRB algorithm: Bound
    
    %Check if lower corners of the two boxes are feasible
    feasible = zeros(2,1); %Set l:th element to one if the l:th box is feasible
    totalNbrOfEvaluations(k) = 0; %Number of convex problems solved in k:th iteration 
    for l = 1:2
        
        %Compute potential performance in upper corner of l:th new box
        if problemMode == 1
            localUpperBoundl = weights'*newBoxesUpperCorners(:,l);
        elseif problemMode == 2
            localUpperBoundl = geomean_weighted(newBoxesUpperCorners(:,l),weights);
        end
        
        %Check if potential performance is better than current best
        %feasible solution
        if localUpperBoundl>cbv
            
            %Check if the current local feasible point is actually in the
            %box (it might not in the current box due to branching and
            %reduction procedures)
            if min(localFeasibleNew(:,l)>=newBoxesLowerCorners(:,l))==1
                %The lower corner is feasible since there we have a local
                %feasible point that strictly dominates this point
                feasible(l) = 1; 
            else
                %The feasibility of the lower corner cannot be determined
                %by previous information and we need to solve a convex
                %feasibility problem.
                gammavar = 2.^(newBoxesLowerCorners(:,l))-1; %Transform lower corner into SINR requirements 
                [checkFeasibility,W] = functionFeasibilityProblem_cvx(H,D,Qsqrt,q,gammavar); %Solve the feasibility problem
                totalNbrOfEvaluations(k) = totalNbrOfEvaluations(k)+1; %Increase number of feasibility evaluations in k:th iteration
                
                %Check if the point was feasible
                if checkFeasibility == false
                    feasible(l) = 0; %Not feasible
                elseif checkFeasibility == true
                    feasible(l) = 1; %Feasible
                    localFeasibleNew(:,l) = newBoxesLowerCorners(:,l); %Update local feasible point
                    
                    %Compute the performance in the lower corner
                    if problemMode==1
                        localLowerBoundl = weights'*newBoxesLowerCorners(:,l);
                    elseif problemMode==2
                        localLowerBoundl = geomean_weighted(newBoxesLowerCorners(:,l),weights);
                    end
                    
                    %If the feasible point is better than all found so far,
                    %then it is stored.
                    if localLowerBoundl>cbv
                        bestFeasible = newBoxesLowerCorners(:,l);
                        cbv = localLowerBoundl;
                        Woptimal = W;
                    end
                end
            end
            
        else
            feasible(l) = 0; %The box whole box is inside the rate region and can be removed
        end

    end
    

    %Search for a better feasible point in the outmost of the new boxes.
    %The search is only done if the box is feasible.
    if feasible(2)==1
        
        %Solve an FPO problem to find a better feasible point the outmost
        %box. We search for a point on the line between the lower and upper
        %corner, with an accuracy given by delta.
        [interval,W,FPOevaluations] = functionFairnessProfile_cvx(H,D,Qsqrt,q,delta,newBoxesLowerCorners(:,2),newBoxesUpperCorners(:,2));
        
        %Update the number of feasibility evaluations
        totalNbrOfEvaluations(k) = totalNbrOfEvaluations(k)+FPOevaluations; 
        
        %The new feasible point found by the FPO problem
        newFeasiblePoint = interval(:,1);
        
        %The point interval(:,2) is either infeasible or on the Pareto
        %boundary. All points in the box that strictly dominate this point
        %can be ignored. The Kr corner points of the remaining polyblock 
        %are calculated and will be used to improve the local upper bound.
        reduced_corners = repmat(newBoxesUpperCorners(:,2),[1 Kr])-diag(newBoxesUpperCorners(:,2)-interval(:,2));
        
        %A local upper bound can be computed as the largest system utility
        %achieved among the Kr corner points computed above. This new bound 
        %replaces the current local upper bound if it is smaller.
        if problemMode == 1
            cubNew(2) = min([max(weights'*reduced_corners) cubNew(2)]);
        elseif problemMode == 2
            cubNew(2) = min([max(geomean_weighted(reduced_corners,weights)) cubNew(2)]);
        end
        
        %Update the local feasible point, if the new point is better.
        %Update the global feasible point, if the new point is better.
        if problemMode == 1
            
            if weights'*newFeasiblePoint > weights'*localFeasibleNew(:,2)
                localFeasibleNew(:,2) = newFeasiblePoint;
            end
            
            if weights'*newFeasiblePoint > cbv
                bestFeasible = newFeasiblePoint;
                cbv = weights'*newFeasiblePoint;
                Woptimal = W; %Store beamforming for current best solution

                % disp(Woptimal)
                % Whistory(kopt,:,:) = Woptimal;
                % kopt = kopt + 1;
            end
            
        elseif problemMode == 2
            
            if geomean_weighted(newFeasiblePoint,weights) > geomean_weighted(localFeasibleNew(:,2),weights)
                localFeasibleNew(:,2) = newFeasiblePoint;
            end
            
            if geomean_weighted(newFeasiblePoint,weights) > cbv
                bestFeasible = newFeasiblePoint;
                cbv = geomean_weighted(newFeasiblePoint,weights);
                Woptimal = W; %Store beamforming for current best solution
            end
            
        end
        
    end

    
    
    
    %Step 4 of BRB algorithm: Prepare for next iteration
    
    %Check which boxes that should be kept for the next iteration
    keep = cub>cbv; %Only keep boxes that might contain better points than current best solution
    keep(ind) = false; %Remove the box that was branched
    keepnew = (feasible==1); %Only keep new boxes that are feasible
    
    %Update the boxes and their local information for the next iteration
    %of the algorithm.
    boxesLowerCorners = [boxesLowerCorners(:,keep) newBoxesLowerCorners(:,keepnew)];
    boxesUpperCorners = [boxesUpperCorners(:,keep) newBoxesUpperCorners(:,keepnew)];
    cub = [cub(keep) cubNew(keepnew)];
    localFeasible = [localFeasible(:,keep) localFeasibleNew(:,keepnew)];
    
    %Store the lower and upper bounds in the k:th iteration to enable
    %plotting of the progress of the algorithm.
    lowerBound(k) = cbv;
    upperBound(k) = max(cub);

    % disp("k = " +num2str(k))
    % disp("lowerBound = " + num2str(lowerBound(k)) + ", upperBound = " + num2str(upperBound(k)))
    
    
    
    if saveBoxes == 1
        boxes{k}.lowerCorners=boxesLowerCorners;
        boxes{k}.upperCorners=boxesUpperCorners;
    end
    
    %Check termination conditions
    if sum(totalNbrOfEvaluations) >= maxFuncEvaluations %Maximal number of feasibility evaluations has been used
        break;
    elseif upperBound(k)-lowerBound(k) <= epsilon %Predefined accuracy of optimal solution has been achieved
        break;
    end
end


%Prepare output by removing parts of output vectors that were not used
totalNbrOfEvaluations = totalNbrOfEvaluations(1:k);
bounds = [lowerBound(1:k) upperBound(1:k)];
% Whistory = Whistory(1:k,:,:);





function y = geomean_weighted(x,w)
%Calculate weighted proportional fairness of each column of z.
%The corresponding weights are given in w

y = prod(x.^repmat(w,[1 size(x,2)]),1);
