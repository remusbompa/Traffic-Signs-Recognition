// Functions that toggle visibility of selected containers
// Created by Johan Wiklund, ISY, LiU
// $Id: toggle.js 8 2009-10-07 10:50:46Z jowi $

function toggleVis(obj) {
    var el = document.getElementById(obj);
    if ( el.style.display != 'none' ) {
	el.style.display = 'none';
    } else {
	el.style.display = 'block';
    }
}

function toggleByClass( searchClass, node, tag ) {
    var classElements = new Array();
    if ( node == null )
	node = document;
    if ( tag == null )
	tag = '*';
    var els = node.getElementsByTagName(tag);
    var elsLen = els.length;
    var pattern = new RegExp("(^|\s)"+searchClass+"(\s|$)");
    for (i = 0, j = 0; i < elsLen; i++) {
	if ( pattern.test(els[i].className) ) {
	    classElements[j] = els[i];
	    j++;
	}
    }
    for (i = 0; i < classElements.length; i++) {
	var el = classElements[i];
	if ( el.style.display != 'none' ) {
	    el.style.display = 'none';
	} else {
	    el.style.display = 'block';
	}
    }
}
