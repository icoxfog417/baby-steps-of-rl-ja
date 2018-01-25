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
                var cell = {
                    index: [rowIndex, columnIndex],
                    attribute: 0,
                    rewards: [0, 0, 0, 0]
                }
                this.grid[rowIndex].push(cell);
            }
        },
        getCellAttribute: function(index){
            var cell = this.grid[index[0]][index[1]];
            switch(cell.attribute){
                case 1:
                    return "treasure"
                case -1:
                    return "danger"
                case 9:
                    return "block"
            }
            if(this.selectedIndex != null && (this.selectedIndex[0] == index[0] && this.selectedIndex[1] == index[1])){
                return "active"
            }
            if(index[0] == (this.grid.length - 1) && index[1] == 0){
                return "agent"
            }
        },
        selectCell: function(index){
            // [row, 0] is Agent point
            if(!(index[0] == (this.grid.length - 1) && index[1] == 0)){
                this.selectedIndex = index;
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
        clearAttribute: function(index){
            this.selectedIndex = index;
            this.setAttribute(0);
            this.selectedIndex = null;
        },
        setAttribute: function(attr){
            var index = this.selectedIndex;
            if(this.selectedIndex != null){
                var cell = this.grid[index[0]][index[1]];
                cell.attribute = attr;
            }
        }
    }
})
