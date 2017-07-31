#include "vpa2cuda.h"

void printUsage(char*);

#define DEBUG if(verbose) std::cout

typedef struct vpaReportRow_t{
    std::string opType, opRetType, op1, op2, opRet;
    int line;
} vpaReportRow;

void normalConversion(std::map<std::string, vpaReportRow> vpaTable, std::map<std::string, int> solutionVector, int lengthVector, bool verbose, bool inlineAssignments);
void compactConversion(std::map<std::string, vpaReportRow> vpaTable, std::map<std::string, int> solutionVector, int lengthVector, bool verbose);

int main(int argc, char* argv[]){
    
    //vpa2cuda -r report -b 3 1 2 3
    if(argc < 5) printUsage(argv[0]);
    
    char opt;
    std::string reportPath;
    std::map<std::string, int> solutionVector;
    int lengthVector;
    int i;
    bool inlineAssignments = false;
    bool compact = false;
    double error, reward, penalty;
    bool verbose = false;
    while ((opt = getopt(argc, argv, "r:b:l:v")) != -1) {
        switch (opt) {
            case 'v':
                verbose = true;
                break;
            case 'r':
                reportPath = std::string(optarg);
                break;
            case 'l':
                if(atoi(optarg) == 1)
                    inlineAssignments = true;
                else if(atoi(optarg) == 2)
                    compact = true;
                else if(atoi(optarg) != 0)
                    printUsage(argv[0]);
                break;
            case 'b':
                error = atof(optarg);
                DEBUG << "Input solution error: " << error << std::endl;
                reward = atof(argv[optind]);
                DEBUG << "Input solution reward: " << reward << std::endl;
                penalty = atof(argv[optind+1]);
                DEBUG << "Input solution penalty: " << penalty << std::endl;
                lengthVector = atoi(argv[optind+2]);
                DEBUG << "Input solution vector length: " << lengthVector << std::endl << "Elements: ";
               // solutionVector = std::map<std::string, int>(lengthVector);
                for (i = 0; i < lengthVector; i++) {
                    solutionVector[std::string("OP_"+std::to_string(i))] = (atoi(argv[optind+i+3]));
                    DEBUG << "OP_" << i << ": " << solutionVector[std::string("OP_"+std::to_string(i))] << "; ";
                }
                DEBUG << std::endl;
                break;
            case 'h':
            default: /* '?' */
                printUsage(argv[0]);
        }
    }
    
    std::ifstream vpa_report(reportPath.c_str());
    
    std::map<std::string, vpaReportRow> vpaTable;
    
    ::std::string getLine, OpId;
    ::std::string line;
    ::std::string OpTy, OpRetTy;
    ::std::string Op1, Op2, OpRet; // Operands
    DEBUG << "Reading the VPA report in CSV format..." << std::endl;
    while (!vpa_report.eof()) { //Retrieve a new row from the CSV until exist
        vpa_report >> getLine;
        char temp[100];
        std::strcpy(temp,getLine.c_str());
        char* tokenized = std::strtok(temp, ",");
        
        OpId = std::string(tokenized);
        tokenized = std::strtok(NULL, ",");
        line = std::string(tokenized);
        tokenized = std::strtok(NULL, ",");
        OpRetTy = std::string(tokenized);
        tokenized = std::strtok(NULL, ",");
        OpTy = std::string(tokenized);
        tokenized = std::strtok(NULL, ",");
        Op1 = std::string(tokenized);
        Op1.erase(0,1);
        Op1.erase(Op1.end()-1, Op1.end());
        tokenized = std::strtok(NULL, ",");
        Op2 = std::string(tokenized);
        Op2.erase(0,1);
        Op2.erase(Op2.end()-1, Op2.end());
        tokenized = std::strtok(NULL, ",");
        OpRet = std::string(tokenized);
        OpRet.erase(0,1);
        OpRet.erase(OpRet.end()-1, OpRet.end());
        DEBUG << vpaTable.size() << " --> " << OpId << " " << line << " " << OpTy << " " << OpRetTy << " " << Op1 << " " << Op2 << " "<<OpRet << std::endl;

        vpaTable[OpId] = (vpaReportRow){OpTy, OpRetTy, Op1, Op2, OpRet, atoi(line.c_str())};
    }
    
    DEBUG << "Elaborating the solution..." << std::endl;
    
    if(!compact)
        normalConversion(vpaTable, solutionVector, lengthVector, verbose, inlineAssignments);
    else
        compactConversion(vpaTable, solutionVector, lengthVector, verbose);
    return EXIT_SUCCESS;
}

void normalConversion(std::map<std::string, vpaReportRow> vpaTable, std::map<std::string, int> solutionVector, int lengthVector, bool verbose, bool inlineAssignments){    
    std::map<std::string, bool> declaredVariables;
    
    std::string declarations = "";
    std::map<std::string, std::string> variableTypes;
    std::vector<std::string> assignment(lengthVector);
    std::string rowAssignment, operatorAssignment1, operatorAssignment2;
    int i;
    for(i = 0; i < lengthVector; i++){
        auto row = *vpaTable.find(std::string("OP_"+std::to_string(i)));
        if(0 != row.second.opRet.compare("NULL")){
            if(declaredVariables.find(row.second.opRet) == declaredVariables.end()){
                declaredVariables[row.second.opRet] = true;
                if(0 == row.second.opRetType.compare("DOUBLE")){
                    if(0 == solutionVector[row.first]){
                        declarations+="double "+row.second.opRet + ";\n";
                        variableTypes[std::string("OP_"+std::to_string(i))] = "DOUBLE";
                    }else if(solutionVector[row.first] == 1){
                        declarations+="float "+row.second.opRet + ";\n";
                        variableTypes[std::string("OP_"+std::to_string(i))] = "FLOAT";
                    }else{
                        declarations+="half "+row.second.opRet + ";\n";
                        variableTypes[std::string("OP_"+std::to_string(i))] = "HALF";
                    }
                } else {
                    if(0 == solutionVector[row.first]){
                        declarations+="float "+row.second.opRet + ";\n";
                        variableTypes[std::string("OP_"+std::to_string(i))] = "FLOAT";
                    }else{
                        declarations+="half "+row.second.opRet + ";\n";
                        variableTypes[std::string("OP_"+std::to_string(i))] = "HALF";
                    }
                }
            }
        } else {
            if(0 == row.second.opRetType.compare("DOUBLE")){
                if(0 == solutionVector[row.first]){
                    declarations+="double "+row.first + ";\n";
                    variableTypes[std::string("OP_"+std::to_string(i))] = "DOUBLE";
                }else if(solutionVector[row.first] == 1){
                    declarations+="float "+row.first + ";\n";
                    variableTypes[std::string("OP_"+std::to_string(i))] = "FLOAT";
                }else{
                    declarations+="half "+row.first + ";\n";
                    variableTypes[std::string("OP_"+std::to_string(i))] = "HALF";
                }
            } else {
                if(0 == solutionVector[row.first]){
                    declarations+="float "+row.first + ";\n";
                    variableTypes[std::string("OP_"+std::to_string(i))] = "FLOAT";
                }else{
                    declarations+="half "+row.first + ";\n";
                    variableTypes[std::string("OP_"+std::to_string(i))] = "HALF";
                }
            }
        }
    }
    DEBUG << "/*Variables Declaration */"<< std::endl << declarations;
    
    int block = 0;
    int currentLine = -1;
    std::string variableTypeOp1, variableTypeOp2;
    for(i = 0; i < lengthVector; i++){
        rowAssignment = operatorAssignment1 = operatorAssignment2 = "";
        
        auto row = *vpaTable.find(std::string("OP_"+std::to_string(i)));
        if(0 != row.second.opRet.compare("NULL")){
            if(!inlineAssignments) block++;
            rowAssignment += row.second.opRet + " = ";
        } else{
            rowAssignment += row.first + " = ";
        }
        
        //get variable name
        if(vpaTable.find(row.second.op1) != vpaTable.end() && 0 != vpaTable[row.second.op1].opRet.compare("NULL")){
            operatorAssignment1 = vpaTable[row.second.op1].opRet;
        } else {
            operatorAssignment1 = row.second.op1;
        }
        
        //get the type
        if(vpaTable.find(row.second.op1) != vpaTable.end()){
            variableTypeOp1 = variableTypes[row.second.op1];
        } else {
            variableTypeOp1 = row.second.opRetType;
        }
        
        //decide the cast
        if(0 != variableTypes[row.first].compare(variableTypeOp1)){
            if(0 == variableTypes[row.first].compare("DOUBLE")){
                operatorAssignment1 = "__float2double(" + operatorAssignment1 + ")";
                if(0 == variableTypeOp1.compare("HALF")){
                    operatorAssignment1 = "__half2float(" + operatorAssignment1 + ")";
                }
            } else if (0 == variableTypes[row.first].compare("FLOAT")){
                if(0 == variableTypeOp1.compare("HALF")){
                    operatorAssignment1 = "__half2float(" + operatorAssignment1 + ")";
                } else {
                    operatorAssignment1 = "__double2float_rn(" + operatorAssignment1 + ")";
                }
            } else {
                if(0 == variableTypeOp1.compare("FLOAT")){
                    operatorAssignment1 = "__float2half(" + operatorAssignment1 + ")";
                }
            }
        }
        
        //get variable name
        if(vpaTable.find(row.second.op2) != vpaTable.end() && 0 != vpaTable[row.second.op2].opRet.compare("NULL")){
            operatorAssignment2 = vpaTable[row.second.op2].opRet;
        } else {
            operatorAssignment2 = row.second.op2;
        }
        
        //get the type
        if(vpaTable.find(row.second.op2) != vpaTable.end()){
            variableTypeOp2 = variableTypes[row.second.op2];
        } else {
            variableTypeOp2 = row.second.opRetType;
        }
        
        //decide the cast
        if(0 != variableTypes[row.first].compare(variableTypeOp2)){
            if(0 == variableTypes[row.first].compare("DOUBLE")){
                operatorAssignment2 = "__float2double(" + operatorAssignment2 + ")";
                if(0 == variableTypeOp2.compare("HALF")){
                    operatorAssignment2 = "__half2float(" + operatorAssignment2 + ")";
                }
            } else if (0 == variableTypes[row.first].compare("FLOAT")){
                if(0 == variableTypeOp2.compare("HALF")){
                    operatorAssignment2 = "__half2float(" + operatorAssignment2 + ")";
                } else {
                    operatorAssignment2 = "__double2float_rn(" + operatorAssignment2 + ")";
                }
            } else {
                if(0 == variableTypeOp2.compare("FLOAT")){
                    operatorAssignment2 = "__float2half(" + operatorAssignment2 + ")";
                }
            }
        }
        
        if(0 == row.second.opType.compare("ADD")){
	       if(0 == variableTypes[row.first].compare("HALF")){
            	rowAssignment += "__hadd("+operatorAssignment1 + ", " + operatorAssignment2 + ")";
	       } else {
            	rowAssignment += operatorAssignment1 + " + " + operatorAssignment2;
	       }
	    } else if(0 == row.second.opType.compare("SUB")){
	       if(0 == variableTypes[row.first].compare("HALF")){
            	rowAssignment += "__hsub("+operatorAssignment1 + ", " + operatorAssignment2 + ")";
	       } else {
            	rowAssignment += operatorAssignment1 + " - " + operatorAssignment2;
	       }
	    } else if(0 == row.second.opType.compare("MUL")){
            if(0 == variableTypes[row.first].compare("HALF")){
            	rowAssignment += "__hmul("+operatorAssignment1 + ", " + operatorAssignment2 + ")";
	       } else {
            	rowAssignment += operatorAssignment1 + " * " + operatorAssignment2;
	       }
	   } else {
	       if(0 == variableTypes[row.first].compare("HALF")){
             	rowAssignment += "hdiv("+operatorAssignment1 + ", " + operatorAssignment2 + ")";
	       } else {
            	rowAssignment += operatorAssignment1 + " / " + operatorAssignment2;
	       }
	   }
       
        inlineAssignments ? rowAssignment += "; " : rowAssignment += ";\n";
        
        if(inlineAssignments && currentLine != row.second.line){
            if(currentLine >= 0) assignment[block] += "/*line "+ std::to_string(currentLine) +" */";
            block++;
            currentLine = row.second.line;
        }

        assignment[block] = rowAssignment + assignment[block];
        
    }
    
    DEBUG << "/*Assignment */"<< std::endl;
    for(i = 0; i < assignment.size(); i++)
        DEBUG << assignment[i];
    
    std::cout << declarations << std::endl;
    
    for(i = 0; i < assignment.size(); i++)
        inlineAssignments ? std::cout << assignment[i] << std::endl : std::cout << assignment[i];
    
}

void compactConversion(std::map<std::string, vpaReportRow> vpaTable, std::map<std::string, int> solutionVector, int lengthVector, bool verbose){
    std::map<std::string, bool> declaredVariables;
    
    std::map<std::string, std::string> variableTypes;
    int i;
    for(i = 0; i < lengthVector; i++){
        auto row = *vpaTable.find(std::string("OP_"+std::to_string(i)));
        if(0 != row.second.opRet.compare("NULL")){
            if(declaredVariables.find(row.second.opRet) == declaredVariables.end()){
                declaredVariables[row.second.opRet] = true;
                if(0 == row.second.opRetType.compare("DOUBLE")){
                    if(0 == solutionVector[row.first]){
                        variableTypes[std::string("OP_"+std::to_string(i))] = "DOUBLE";
                    }else if(solutionVector[row.first] == 1){
                        variableTypes[std::string("OP_"+std::to_string(i))] = "FLOAT";
                    }else{
                        variableTypes[std::string("OP_"+std::to_string(i))] = "HALF";
                    }
                } else {
                    if(0 == solutionVector[row.first]){
                        variableTypes[std::string("OP_"+std::to_string(i))] = "FLOAT";
                    }else{
                        variableTypes[std::string("OP_"+std::to_string(i))] = "HALF";
                    }
                }
            }
        } else {
            if(0 == row.second.opRetType.compare("DOUBLE")){
                if(0 == solutionVector[row.first]){
                    variableTypes[std::string("OP_"+std::to_string(i))] = "DOUBLE";
                }else if(solutionVector[row.first] == 1){
                    variableTypes[std::string("OP_"+std::to_string(i))] = "FLOAT";
                }else{
                    variableTypes[std::string("OP_"+std::to_string(i))] = "HALF";
                }
            } else {
                if(0 == solutionVector[row.first]){
                    variableTypes[std::string("OP_"+std::to_string(i))] = "FLOAT";
                }else{
                    variableTypes[std::string("OP_"+std::to_string(i))] = "HALF";
                }
            }
        }
    }
}

void printUsage(char* name){
    std::cout << "Usage: " << name << " [-v] -l<0|1|2> -r <path/to/the/report> -b <length> <solution vector>" << std::endl;
    std::cout.flush();
    exit(EXIT_FAILURE);
}
