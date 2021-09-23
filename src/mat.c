#include "tensor.h"

#include <math.h>
#include <memory.h>

void Mat::creat(int _w, int _elemsize)
{    
    release();

    dims = 1;
    w = _w;
    size = w;
    elemsize = _elemsize;

    data = (unsigned char*)malloc(w * elemsize);
    memset(data, 0, w * elemsize);
}

void Mat::creat(int _h, int _w, int _elemsize)
{    
    release();

    dims = 2;
    h = _h;
    w = _w;
    size = h * w;
    elemsize = _elemsize;

    data = (unsigned char*)malloc(h * w * elemsize);
    memset(data, 0, h * w * elemsize);
}

void Mat::creat(int _h, int _w, int _cin, int _elemsize)
{    
    release();

    dims = 3;
    h = _h;
    w = _w;
    cin = _cin;
    cout = _cin;
    size = h * w * cin;
    elemsize = _elemsize;

    data = (unsigned char*)malloc(h * w * cin * elemsize);
    memset(data, 0, h * w * cin * elemsize);
}

void Mat::creat(int _h, int _w, int _cin, int _cout, int _elemsize)
{
    release();

    dims = 4;
    h = _h;
    w = _w;
    cin = _cin;
    cout = _cout;
    batch = _cout;
    size = h * w * cin * cout;
    elemsize = _elemsize;

    data = (unsigned char*)malloc(h * w * cin * cout * elemsize);
    memset(data, 0, h * w * cin * cout * elemsize);
}

std::ostream & operator<<(std::ostream & out, Mat& m)
{
    switch (m.dims)
    {
    case 0:
        return out;
    case 1:
        return out << "(" << m.w << ")";
    case 2:
        return out << "(" << m.h << ", " << m.w << ")";
    case 3:
        return out << "(" << m.h << ", " << m.w << ", " << m.cin << ")";
    case 4:
        return out << "(" << m.h << ", " << m.w << ", " << m.cin << ", " << m.cout << ")";
    default:
        return out;
    }
}