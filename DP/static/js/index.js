Vue.config.debug = true;
var app = new Vue({
    el: "#app",
    delimiters: ["[[", "]]"],
    data: {
        row: 3,
        column: 4,
        grid: [],
        selectedIndex: null,
        simulation: false
    },
    created: function(){
        this.draw();
        this.selectedIndex = [0, 3];
        this.setTreasure();
        this.selectedIndex = [1, 3];
        this.setDanger();
        this.selectedIndex = [1, 1];
        this.setBlock();
    },
    methods: {
        init: function(){
            this.selectedIndex = null;
        },
        draw: function(){
            this.init();
            this.makeGrid();
        },
        makeGrid: function(){
            this.grid = [];
            var size = this.row * this.column;
            for(var i = 0; i < size; i++){
                var rowIndex = Math.floor(i / this.column);
                var columnIndex = i % this.column;
                if(columnIndex == 0){
                    this.grid.push([]);
                }
                var cellAttribute = 0;
                this.grid[rowIndex].push(cellAttribute);
            }
        },
        getCellAttribute: function(row, column){
            var attribute = this.grid[row][column];
            switch(attribute){
                case 1:
                    return "treasure"
                case -1:
                    return "danger"
                case 9:
                    return "block"
            }
            if(this.selectedIndex != null && (this.selectedIndex[0] == row && this.selectedIndex[1] == column)){
                return "active"
            }
            if(row == (this.grid.length - 1) && column == 0){
                return "agent"
            }
        },
        simulate: function(){
            var data = {
                "grid": this.grid
            }
            fetch("/simulate", {
                method: "POST",
                credentials: "same-origin",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(data)
            }).then(function(resp){
                console.log(resp)
            })
        },
        selectCell: function(row, column){
            // [row, 0] is Agent point
            if(!(row == (this.grid.length - 1) && column == 0)){
                this.selectedIndex = [row, column];
            }
        },
        setTreasure: function(){
            this.setAttribute(1);
        },
        setDanger: function(){
            this.setAttribute(-1);
        },
        setBlock: function(){
            this.setAttribute(9);
        },
        clearAttribute: function(row, column){
            this.selectedIndex = [row, column];
            this.setAttribute(0);
        },
        setAttribute: function(attribute){
            var index = this.selectedIndex;
            if(this.selectedIndex != null){
                this.grid[index[0]][index[1]] = attribute;
                this.selectedIndex = null;
            }
        }
    }
})
