(module
  (type (;0;) (func (param i32 i32) (result i32)))
  (type (;1;) (func (param i64)))
  (import "env" "gas_counter" (global (;0;) (mut i64)))
  (func $add_locals (;0;) (type 0) (param $x i32) (param $y i32) (result i32)
    (local $t i32)
    i64.const 5
    call 2
    local.get $x
    local.get $y
    call $add
    local.set $t
    local.get $t
  )
  (func $add (;1;) (type 0) (param $x i32) (param $y i32) (result i32)
    i64.const 3
    call 2
    local.get $x
    local.get $y
    i32.add
  )
  (func (;2;) (type 1) (param i64)
    global.get 0
    local.get 0
    i64.sub
    global.set 0
    global.get 0
    i64.const 0
    i64.lt_s
    if ;; label = @1
      unreachable
    end
  )
)