#include "vpa2cuda.h"

void printUsage(char*);

#define DEBUG if(verbose) std::cout

typedef struct vpaReportRow_t{
    std::string opType, opRetType, op1, op2, opRet;
    int line;
} vpaReportRow;

int main(int argc, char* argv[]){
    
    //vpa2cuda -r report -b 3 1 2 3
    if(argc < 5) printUsage(argv[0]);
    
    char opt;
    std::string reportPath;
    std::map<std::string, int> solutionVector;
    int lengthVector;
    int i;
    bool inlineAssignments = false;
    double error, reward, penalty;
    bool verbose = false;
    while ((opt = getopt(argc, argv, "r:b:vl")) != -1) {
        switch (opt) {
            case 'v':
                verbose = true;
                break;
            case 'r':
                reportPath = std::string(optarg);
                break;
            case 'l':
                inlineAssignments = true;
                break;
            case 'b':
                error = atof(optarg);
                DEBUG << "Solution error: " << error << std::endl;
                reward = atof(argv[optind]);
                DEBUG << "Solution reward: " << reward << std::endl;
                penalty = atof(argv[optind+1]);
                DEBUG << "Solution penalty: " << penalty << std::endl;
                lengthVector = atoi(argv[optind+2]);
                DEBUG << "Solution vector length: " << lengthVector << std::endl << "Elements: ";
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
        DEBUG << OpId << " " << line << " " << OpTy << " " << OpRetTy << " " << Op1 << " " << Op2 << " "<<OpRet << std::endl;

        vpaTable[OpId] = (vpaReportRow){OpTy, OpRetTy, Op1, Op2, OpRet, atoi(line.c_str())};
    }
    
    std::map<std::string, bool> declaredVariables;
    
    std::string declarations = "";
    std::map<std::string, std::string> variableTypes;
    std::vector<std::string> assigment(lengthVector);
    std::string rowAssignment, operatorAssignment1, operatorAssignment2;
    for(i = 0; i < lengthVector; i++){
        //rowAssignment = "";
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
            //rowAssignment += row.second.opRet + " = ";
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
            //rowAssignment += row.first + " = ";
        }
    }
    DEBUG << "/*Variables Declaration */"<< std::endl << declarations;
    
    int block = 0;
    int currentLine = -1;
    std::string variableType;
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
            variableType = variableTypes[row.second.op1];
        } else {
            variableType = row.second.opRetType;
        }
        
        //decide the cast
        if(0 != variableTypes[row.first].compare(variableType)){
            if(0 == variableTypes[row.first].compare("DOUBLE")){
                operatorAssignment1 = "__float2double(" + operatorAssignment1 + ")";
                if(0 == variableType.compare("HALF")){
                    operatorAssignment1 = "__half2float(" + operatorAssignment1 + ")";
                }
            } else if (0 == variableTypes[row.first].compare("FLOAT")){
                if(0 == variableType.compare("HALF")){
                    operatorAssignment1 = "__half2float(" + operatorAssignment1 + ")";
                } else {
                    operatorAssignment1 = "__double2float_rn(" + operatorAssignment1 + ")";
                }
            } else {
                if(0 == variableType.compare("FLOAT")){
                    operatorAssignment1 = "__float2half_rn(" + operatorAssignment1 + ")";
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
            variableType = variableTypes[row.second.op2];
        } else {
            variableType = row.second.opRetType;
        }
        
        //decide the cast
        if(0 != variableTypes[row.first].compare(variableType)){
            if(0 == variableTypes[row.first].compare("DOUBLE")){
                operatorAssignment2 = "__float2double(" + operatorAssignment2 + ")";
                if(0 == variableType.compare("HALF")){
                    operatorAssignment2 = "__half2float(" + operatorAssignment2 + ")";
                }
            } else if (0 == variableTypes[row.first].compare("FLOAT")){
                if(0 == variableType.compare("HALF")){
                    operatorAssignment2 = "__half2float(" + operatorAssignment2 + ")";
                } else {
                    operatorAssignment2 = "__double2float_rn(" + operatorAssignment2 + ")";
                }
            } else {
                if(0 == variableType.compare("FLOAT")){
                    operatorAssignment2 = "__float2half_rn(" + operatorAssignment2 + ")";
                }
            }
        }
        
        if(0 == row.second.opType.compare("ADD"))
            rowAssignment += operatorAssignment1 + " + " + operatorAssignment2;
        else if(0 == row.second.opType.compare("SUB"))
            rowAssignment += operatorAssignment1 + " - " + operatorAssignment2;
        else if(0 == row.second.opType.compare("MUL"))
            rowAssignment += operatorAssignment1 + " * " + operatorAssignment2;
        else
            rowAssignment += operatorAssignment1 + " / " + operatorAssignment2;
        
       
        inlineAssignments ? rowAssignment += "; " : rowAssignment += ";\n";
        
        if(inlineAssignments && currentLine != row.second.line){
            if(currentLine >= 0) assigment[block] += "/*line "+ std::to_string(currentLine) +" */";
            block++;
            currentLine = row.second.line;
        }

        assigment[block] = rowAssignment + assigment[block];
        
    }
    
    DEBUG << "/*Assignment */"<< std::endl;
    for(i = 0; i < assigment.size(); i++)
        DEBUG << assigment[i];
    
    std::cout << declarations << std::endl;
    
    for(i = 0; i < assigment.size(); i++)
        inlineAssignments ? std::cout << assigment[i] << std::endl : std::cout << assigment[i];
    
    
    return EXIT_SUCCESS;
}

void printUsage(char* name){
    std::cout << "Usage: " << name << " [-v] -r <path/to/the/report> -b <length> <solution vector>" << std::endl;
    std::cout.flush();
    exit(EXIT_FAILURE);
}
