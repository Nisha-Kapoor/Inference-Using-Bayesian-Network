import copy
import collections
from decimal import Decimal
import numpy as np

#method is used to add a key value pair to the front of an ordered dictionary
def ordered_dict_prepend(dct, key, value, dict_setitem=dict.__setitem__):
    root = dct._OrderedDict__root
    first = root[1]

    if key in dct:
        link = dct._OrderedDict__map[key]
        link_prev, link_next, _ = link
        link_prev[1] = link_next
        link_next[0] = link_prev
        link[0] = root
        link[1] = first
        root[1] = first[0] = link
    else:
        root[1] = first[0] = dct._OrderedDict__map[key] = [root, first, key]
        dict_setitem(dct, key, value)


#put the queries in a list 
file=open("input.txt", "r")
firstLine=file.readline()
query=[]
while firstLine.strip() != "******" :
    query.append(firstLine)
    firstLine=file.readline()

#Forming the bayesian network
decision=[]
bn=collections.OrderedDict()
bayesNode=file.readline()
while bayesNode:
    dict={}
    parents=[]
    nodeLine=bayesNode.strip().split(" ")

    var=nodeLine[0]
    bn[var]={}

    parents=[]

    if (len(nodeLine)>1):
         parents= nodeLine[2:]
    bn[var]["parent"]=parents
    cptLine=file.readline().strip()

    if (cptLine=="decision"):
        bn[var]["prob"]=1.0
        decision.append(var)
        bayesNode = file.readline()
    elif not bn[var]["parent"]:

        bn[var]["prob"]=float(cptLine)
        bayesNode = file.readline()

    else:
        dict={}
        bn[var]["prob"] = -1.0
        while cptLine and cptLine.strip()!="***" and cptLine.strip()!="******":
            if "+" in cptLine or "-" in cptLine:
                table=cptLine.strip().split(" ")

                key=" ".join(table[1:])
                dict[key]=float(table[0])
            cptLine=file.readline();

    bn[var]["cpt"]=dict

    bayesNode=file.readline()



def elimination_ask(X, e, bn):
    Q=[]
    factors = []
    for var in reversed(getVar(bn)):
        factors.append(makeFactor(var, e, bn))
        if is_hidden(var, X, e):
            factors = sumOutVars(var, factors, bn)
    f= pointwiseProduct(factors, bn).cpt
    for i in f:
        Q.append(f[i])
    normalize(Q)
    return Q


def is_hidden(var, X, e):
    return var not in X and var not in e


def makeFactor(var, e, bn):

    node = bn[var]

    variables = [X for X in [var] + node["parent"] if X not in e]
    cpt = {eventVals(e1, variables): condProb(var,e1)
           for e1 in events(variables, bn, e)}

    return Factors(variables, cpt)


def pointwiseProduct(factors, bn):
    return reduce(lambda f, g: f.pointwiseProduct(g, bn), factors)


def sumOutVars(var, factors, bn):

    result, var_factors = [], []
    for f in factors:
        (var_factors if var in f.variables else result).append(f)
    result.append(pointwiseProduct(var_factors, bn).sumOutVars(var, bn))
    return result




class Factors:

    def __init__(self, variables, cpt):
        self.variables = variables
        self.cpt = cpt

    def pointwiseProduct(self, other, bn):

        variables = list(set(self.variables) | set(other.variables))
        cpt = {eventVals(e, variables): self.p(e) * other.p(e)
               for e in events(variables, bn, {})}
        return Factors(variables, cpt)

    def sumOutVars(self, var, bn):
        variables = [X for X in self.variables if X != var]
        cpt = {eventVals(e, variables): sum(self.p(extend(e, var, val))
                                               for val in ["+","-"])
               for e in events(variables, bn, {})}
        return Factors(variables, cpt)
    def p(self, e):
        return self.cpt[eventVals(e, self.variables)]



def enumeration_ask(X, e, bn):
    Q = [1,4]
    for xi in [True,False]:
        vars=getVar(bn)
        if xi:
            Q[0] = enumerate_all(vars,extend(e, X, "+"))
        else:
            Q[1] = enumerate_all(vars,extend(e, X, "-"))
    return Q



def enumerate_all(vars, e):
    if not vars:
        return 1.0
    y=vars[0]
    if y in e:
        return condProb(y,e) * enumerate_all(vars[1:],e)
    else:
        summation = []

    for j in ["+","-"]:
        e1 = extend(e, y, j)
        summation.append(condProb(y,e1) * enumerate_all(vars[1:],e1))
    return sum(summation)




#Find the probabiliy of the node from Bayesian Network given evidence
def condProb(var, e):
    if bn[var]["prob"]!=-1:
        prob= bn[var]["prob"]
    else:
        temp = []
        for i in bn[var]["parent"]:
            if i in decision and i not in e:
                return 1
            temp.append(e[i])
        key = " ".join(temp)
        #print "Here is the error %s"%var
        if key in bn[var]["cpt"]:
            prob= bn[var]["cpt"][key]
        else:
            #print "modified %s" % modBN
            prob = modBN[var]["cpt"][key]
    if (e[var]=="+"):
        return prob
    else:
        return 1.0-prob



def normalize(Q):
    s = sum(Q)
    for i in range (0, len(Q)):
        Q[i] = Q[i]/ s


def events(variables, bn, e):
    if not variables:
        yield e
    else:
        X, rest = variables[0], variables[1:]
        for e1 in events(rest, bn, e):
            for x in ["+","-"]:
                yield extend(e1, X, x)

def extend(array, variable, values):
    s = copy.deepcopy(array)
    s[variable]=values
    return s

#Get the Bayes Net variables
def getVar(bn):
    vars=[]
    for key in bn:
        if key not in decision:
            vars.append(key)
    return vars


def eventVals(event, variables):
    if isinstance(event, tuple) and len(event) == len(variables):
        return event
    else:
        return tuple([event[var] for var in variables])

'''Call function elimination_ask
P(A="+",B="-"|C="-")=P(B="-"|C="-")*P(A="+"|B="-",C="-")'''
def prep(input, evidence, bn, andflag):
    if andflag:
        ans = []
        for key, value in reversed(list(input.items())):
            inp={}
            inp[key]=value
            answer=(elimination_ask(inp, evidence, bn))
            '''both + and - values are returned
            select the one which applied according to the input'''
            for i in inp:
                if inp[i] == "+":
                    ans.append(answer[0])
                else:
                    ans.append(answer[1])
            ordered_dict_prepend(evidence,key,value)
        product = np.product(ans)
        return Decimal(product)
    else:
        answer = elimination_ask(input, evidence, bn)
        for i in input:
            if input[i] == "+":
                result=answer[0]
            else:
                result=answer[1]
            return Decimal(result)



modBN={}
'''method to calculate EU queries
EU(I="+"|L="-")
'''
def calcEU(test, evidence, bn):
    global modBN
    sign={}
    modBN= copy.deepcopy(bn)
    # the query and evidence vaiables are used as evidence
    for key, value in reversed(list(test.items())):
        if key in decision and key in bn["utility"]["parent"]:
            sign[key]=value
        ordered_dict_prepend(evidence,key, value)
    #parents of utility node are passed as the query varaibles
    parents=copy.deepcopy(bn["utility"]["parent"])
    flag=0
    parentIndices={}
    for p in parents:
        parentIndices[p] = parents.index(p)
    indices={} 
    ''' Do not consider decision node parents
        For this cpt has to be modified 
    '''
    if bool(sign):
        for s in sign:
            parents.remove(s)
            flag=1
            indices[parentIndices[s]]=sign[s]

        cpt = modifyCPT(indices)
        modBN["utility"]["cpt"] = cpt
    length=len(parents)
    andflag=True
    if length==1:
        andflag=False
    results={}
    # the values are calculated for each value of the input variables
    # i.e the utility parents nodes
    for i in range(0, pow(2, length)):
        s = ""
        delim=""
        bin = '{0:0{l}b}'.format(i, l=length)
        #testy stores the particular value combinations of the utlity parent nodes 
        #for that iteration
        testy={}
        for b in range(0, length):
            if bin[b] == "1":
                testy[parents[b]] = "+"
                s=s+delim+"+"
                delim=" "
            elif bin[b] == "0":
                testy[parents[b]] = "-"
                s=s+delim+"-"
                delim=" "
        evid=copy.deepcopy(evidence)
        interm=float(prep(testy, evid, bn, andflag))
        if flag==1:
            results[s] = modBN["utility"]["cpt"][s] * interm
        else:
            results[s] = bn["utility"]["cpt"][s]*interm
    final=sum(results.values())
    return final


#modifies the original cpt table to include values after fixing the value of one of the variables
def modifyCPT(indices):
    cpt = bn["utility"]["cpt"]
    newCPT = {}
    for key in cpt:
        flag = 0
        temp=key.split(" ")
        fixed=key.split(" ")
        for i in indices:
            del temp[i]
            if fixed[i] != indices[i]:
                flag=1
                break
        if flag==0:
            newCPT[" ".join(temp)]=cpt[key]
    return newCPT


#Query Processing parse input and query variables
answers=[]
for q in query:
    # andflag specifies if there are more than one query terms
    andflag = False
    if q[0]=='P':
        test1 = []
        given = []
        input1 = q[2:(len(q) - 2)]
        evidence = collections.OrderedDict()
        test = collections.OrderedDict()
        if "|" in input1:
            input2=input1.split(" | ")
            if "," in input2[1]:
                given=input2[1].split(", ")
            else:
                given.append(input2[1])
            for g in given:
                evidence[g[0]]=g[4]
            if "," in input2[0]:
                test1=input2[0].split(", ")
                andflag=True
            else:
                test1.append(input2[0])
                andflag=False
            for t in test1:
                test[t[0]]=t[4]
        else:
            andflag = True
            if "," in input1:
                test1 = input1.split(", ")
                andflag=True
            else:
                test1.append(input1)
                andflag=False
            for t in test1:
                test[t[0]]=t[4]
        result= prep(test, evidence, bn, andflag)
        res= round(result,10)
        #round of result to 2 decimal places
        answers.append( "%.2f" % float(int(float(res) * 100 + 0.5) / 100.0))

    if q[0]=='E':
        test1 = []
        given = []
        input1=q[3:(len(q) - 2)]
        evidence = collections.OrderedDict()
        test = {}
        if "|" in input1:
            input2 = input1.split(" | ")
            if "," in input2[1]:
                given = input2[1].split(", ")
            else:
                given.append(input2[1])
            for g in given:
                evidence[g[0]] = g[4]
            if "," in input2[0]:
                test1 = input2[0].split(", ")
            else:
                test1.append(input2[0])
            for t in test1:
                test[t[0]] = t[4]
        else:
            if "," in input1:
                test1 = input1.split(", ")
            else:
                test1.append(input1)
            for t in test1:
                test[t[0]] = t[4]
        result=calcEU(test, evidence, bn)
        answers.append(int(round(result,0)))

    if q[0]=='M':
        test1 = []
        given = []
        input1=q[4:(len(q) - 2)]
        evidence = collections.OrderedDict()
        test = {}
        if "|" in input1:
            input2 = input1.split(" | ")
            if "," in input2[1]:
                given = input2[1].split(", ")
            else:
                given.append(input2[1])
            for g in given:
                evidence[g[0]] = g[4]
            if "," in input2[0]:
                test1 = input2[0].split(", ")
            else:
                test1.append(input2[0])
        else:
            if "," in input1:
                test1 = input1.split(", ")
            else:
                test1.append(input1)
        length=len(test1)
        results={}

        # Generate all binary combinations of length number of bits 
        #to get true false assignment of variables
        for i in range(0, pow (2,length)):
            evid = copy.deepcopy(evidence)
            s = ""
            bin = '{0:0{l}b}'.format(i, l=length)
            for b in range (0,length):
                if bin[b]=="1":
                    test[test1[b]]="+"
                    s+="+ "
                elif bin[b]=="0":
                    test[test1[b]] = "-"
                    s+="- "
            results[s]=calcEU(test, evid, bn)
        max=float("-inf")
        for r in results:
            if results[r]>max:
                max=results[r]
                sign=r

        max=int(round(max, 0))
        result=""
        result=sign+str(max)
        answers.append(result)



#write the answers to output.txt
f = open("output.txt", "w")
for a in answers:
    f.write(str(a)+ "\n")
f.close()