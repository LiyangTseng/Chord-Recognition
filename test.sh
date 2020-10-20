function int_formatting(){
    echo $1
    case
        expr ($1 \>= 1) &&  expr ($1 \<= 9) )
        
    esac
}

int_formatting testing